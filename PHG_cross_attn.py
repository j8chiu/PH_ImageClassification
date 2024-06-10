from torch import optim
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np

from torch.nn.utils import clip_grad_norm_
from pytorch_pretrained_vit import ViT
import os
from models.pd_encoder import PersistenceDiagramEncoder 


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1, add_norm=True):
        super().__init__()
        self.add_norm = add_norm

        self.fc_liner = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            # nn.Dropout(p=dropout),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(p=dropout),
        )

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-6)

    def forward(self, x):
        out = self.fc_liner(x)
        if self.add_norm:
            return self.LayerNorm(x + out)
        return out


# class CrossAttention(nn.Module):
#     def __init__(self, dim=768, num_heads=8, 
#                  qkv_bias=False, 
#                  qk_scale=None, 
#                  attn_drop=0.,
#                  proj_drop=0.):
#         super().__init__()
#         # cross attention of (q: cls token and kv: whole sequence)

#         self.num_heads = num_heads
#         head_dim = dim // num_heads
        
#         self.scale = qk_scale or head_dim ** -0.5

#         self.wq = nn.Linear(dim, dim, bias=qkv_bias)
#         self.wk = nn.Linear(dim, dim, bias=qkv_bias)
#         self.wv = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self,q,kv,mask=None): 
#         B, N, C = q.shape #bs, num_patches+1, E, 
#         q = self.wq(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

#         k = self.wk(kv).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
#         v = self.wv(kv).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)

#         attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN(C/H) @ BH1(C/H) -> BHN1
#         #print('attn shape',attn.shape)

#         if mask is not None:
#             mask = mask[:,None,None,:]
#             attn -= 1000.0*(1.0-mask)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # (BHN1 @ BH1(C/H)) -> BHN(C/H) -> BNH(C/H) -> BNC
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x # batch_size, num_patches, 768



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, 
                 qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        # cross attention of q: cls token and kv: whole sequence

        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None): 
        # x = (pd token, image tokens)
        B, N, C = x.shape #bs, num_patches+1, E, 
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        if mask is not None:
            mask = mask[:,None,None,:]
            attn -= 1000.0*(1.0-mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)

        return x # N x 1 x 768
    

class CrossPHGBlock(nn.Module):
    def __init__(self, topo_embed=1024,
                 embed_size=768, 
                 num_heads=12, 
                 norm_layer=nn.LayerNorm,
                 self_attn_model=None,
                 has_mlp = True,
                 curr_layer=0):
        super().__init__()

        self.has_mlp = has_mlp

        self.topo_proj = nn.Linear(topo_embed,embed_size)
        self.invs_topo_proj = nn.Linear(embed_size,topo_embed)

        self.norm1 = norm_layer(embed_size)
        self.cross_attn = CrossAttention(dim=embed_size,
                              num_heads=num_heads, 
                              proj_drop=0.1)
        
        self.self_attn = self_attn_model.transformer.blocks[curr_layer].eval()
        self.norm2 = norm_layer(embed_size)

        if self.has_mlp:
            self.ffn = FeedForward(emb_size=embed_size,hidden_size=embed_size*4)

    def forward(self,img_feats,topo_feats,mask=None):
        # self_attention
        img_feats = self.self_attn(self.norm1(img_feats),mask=mask) # N， num_patches + 1, 768
        topo_feats = self.topo_proj(topo_feats) # N, 1, 768

        img_tokens = img_feats[:,1:,:] # N, num_patches, 768
        img_cls = img_feats[:,0:1,:]

        tmp = torch.concat((topo_feats,img_tokens),dim=1) # N， num_patches + 1, 768
        fusion_cls = self.cross_attn(tmp) # N x 1 x 768
        
        if self.has_mlp:
            fusion_cls = img_cls + self.ffn(self.norm2(fusion_cls)) # N x 1 x 768

        img_feats = torch.concat((fusion_cls,img_tokens),dim=1)
        topo_feats = self.invs_topo_proj(topo_feats) # N x 1 x 1024

        return img_feats,topo_feats # N, num_patches, E

class CrossPHGNet(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        topo_embed = 1024,
        pd_dim = 4,
        alpha = 0.1,
        num_heads=12,
        img_size = 224,
        norm_layer = nn.LayerNorm,
        device='cuda',
        depth = 12,
        num_classes = 7,
        has_mlp = True,
    ):
        super().__init__()

        self.device = device
        self.alpha = alpha
        # Image encoder specifics
        # ViT default patch embeddings
        self.vit = ViT('B_16_imagenet1k', pretrained=True,image_size=img_size).to(self.device) # construct and load 
        self.vit.fc = None
        freeze_model(self.vit)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim,dtype=torch.float32))

        # PD encoder specifics
        self.pd_encoder = PersistenceDiagramEncoder(input_dim = pd_dim)

        self.fusion = nn.ModuleList([
                CrossPHGBlock(topo_embed=topo_embed,
                              self_attn_model = self.vit,
                              embed_size=embed_dim, 
                              num_heads=num_heads, 
                              norm_layer=norm_layer,
                              has_mlp = has_mlp,
                              curr_layer=curr_layer) for curr_layer in range(depth)])

        
        #TODO: Head
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.topo_head = nn.Linear(topo_embed,num_classes)

        self.ce_loss = nn.CrossEntropyLoss()
        

    def forward(self,img,pd,mask=None):
        '''
        @param img: (N, 3, 224, 224)
        @od        
        '''
        N,_,H,W = img.shape

        #print('input shape: ',img.shape)
  
        img = self.vit.patch_embedding(img)
        img_feats = img.flatten(2).transpose(1, 2) # b,gh*gw,d

        img_feats = torch.cat((self.cls_token.expand(N, -1, -1), img_feats), dim=1) # b,num_patches+1,d
        #print('patches shape: ',out.shape)
        img_feats = self.vit.positional_embedding(img_feats)

        pd_feats = self.pd_encoder(pd).unsqueeze(1) # N x 1 x 1024
        

        for blk in self.fusion:
            img_feats,pd_feats = blk(img_feats=img_feats,topo_feats=pd_feats,mask=mask)
            

        cls_out = self.cls_head(out[:,0,:]) # N, num_class
        pd_out = self.topo_head(pd_feats.squeeze(1)) # N, num_class  
        
        return cls_out,pd_out








