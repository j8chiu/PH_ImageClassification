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
from pd_encoder import PersistenceDiagramEncoder 

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1, add_norm=True):
        super().__init__()
        self.add_norm = add_norm

        self.fc_liner = nn.Sequential(
            nn.Linear(emb_size, hidden_size).bfloat16(),
            nn.GELU(),
            # nn.Dropout(p=dropout),
            nn.Linear(hidden_size, emb_size).bfloat16(),
            nn.Dropout(p=dropout),
        )

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-6).bfloat16()

    def forward(self, x):
        out = self.fc_liner(x)
        if self.add_norm:
            return self.LayerNorm(x + out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        # cross attention of (q: cls token and kv: whole sequence)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias).bfloat16()
        self.wk = nn.Linear(dim, dim, bias=qkv_bias).bfloat16()
        self.wv = nn.Linear(dim, dim, bias=qkv_bias).bfloat16()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim).bfloat16()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,q,kv,mask=None): 
        B, N, C = q.shape #bs, num_patches+1, E, 
        q = self.wq(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        k = self.wk(kv).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        v = self.wv(kv).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN(C/H) @ BH1(C/H) -> BHN1
        print('attn shape',attn.shape)

        if mask is not None:
            mask = mask[:,None,None,:].bfloat16()
            attn -= 1000.0*(1.0-mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # (BHN1 @ BH1(C/H)) -> BHN(C/H) -> BNH(C/H) -> BNC
        x = self.proj(x)
        x = self.proj_drop(x)

        return x # batch_size, num_patches, 768


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

        self.topo_proj = nn.Linear(topo_embed,embed_size).bfloat16()
        self.norm1 = norm_layer(embed_size).bfloat16()
        self.cross_attn = CrossAttention(embed_size=embed_size,
                              num_heads=num_heads, 
                              dropout=0.1)
        
        self.self_attn = self_attn_model.transformer.blocks[curr_layer].eval()
        self.norm2 = norm_layer(embed_size).bfloat16()

        if self.has_mlp:
            self.ffn = FeedForward(emb_size=embed_size,hidden_size=embed_size*4)

    def forward(self,img_feats,topo_feats,mask=None):
        # self_attention
        img_feats = self.self_attn(self.norm1(img_feats),mask=mask)
        topo_feats = self.topo_proj(topo_feats)
        img_feats = self.cross_attn(q=img_feats,kv=topo_feats)
        
        if self.has_mlp:
            x = x + self.ffn(self.norm2(x))

        return x # N, num_patches, E

class CrossPHGNet(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        pd_dim = 4,
        num_heads=12,
        mlp_ratio=4,
        norm_layer = nn.LayerNorm,
        device='cuda',
        depth = 12,
        num_classes = 7
    ):
        super().__init__()

        self.device = device
        # Image encoder specifics
        # ViT default patch embeddings
        self.vit = ViT('B_16_imagenet1k', pretrained=True).to(torch.bfloat16).to(self.device) # construct and load 
        self.vit.fc = None
        freeze_model(self.vit)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim,dtype=torch.bfloat16))

        # PD encoder specifics
        self.pd_encoder = PersistenceDiagramEncoder(input_dim = pd_dim)

        self.fusion = nn.ModuleList([
                CrossPHGBlock(topo_embed=1024,
                            embed_size=768, 
                            num_heads=12, 
                            norm_layer=nn.LayerNorm,
                            self_attn_model=self.vit,
                            has_mlp = True,
                            curr_layer=curr_layer) for curr_layer in range(depth)])

        
        #TODO: Head
        self.cls_head = nn.Linear(embed_dim, num_classes).bfloat16()

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,img,pd,mask=None):
        '''
        @param img: (N, 3, 224, 224)
        @od        
        '''
        N,_,H,W = img.shape

        print('input shape: ',img.shape)
  
        img = self.vit.patch_embedding(img)
        out = img.flatten(2).transpose(1, 2) # b,gh*gw,d

        out = torch.cat((self.vit.class_token.expand(N, -1, -1), out), dim=1) # b,num_patches,d
        print('patches shape: ',out.shape)
        out = self.vit.positional_embedding(out)

        pd_feats = self.pd_encoder(pd) # N x 1024

        for blk in self.fusion:
            out = blk(img_feats=out,topo_feats=pd_feats,mask=mask)

        cls_out = self.cls_head(out[:,0,:]) # N, num_class
        
        return cls_out



