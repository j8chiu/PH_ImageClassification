# stdlib
import argparse
from argparse import RawTextHelpFormatter
from tqdm import tqdm
import os
from pathlib import Path

from torchvision import transforms
import torch
import torch.nn as nn
#from timm.utils import accuracy
import numpy as np
import time
import datetime
from torch import optim

# dataset
from dataset.isic_dataset import ISICDataset
# from dataset.prostate_dataset import ProstateDataset
# from dataset.glaucoma_dataset import GlaucomaDataset
# from dataset.mammography_dataset import MammographyDataset
from torch.utils.data.dataloader import DataLoader

# model
from PHG_cross_attn import CrossPHGNet,AllAttnPHGNet
from pd_baseline import collate_fn, compute_accuracy,init_weights



def load_model(args):
    if args.model_name == 'crossPHG':
        model = CrossPHGNet(
            alpha=args.alpha,
            embed_dim=768,
            topo_embed = 1024,
            pd_dim = 4,
            num_heads=12,
            fuse_freq=args.fuse_freq,
            fusion_type = 'cross_attn',
            norm_layer = nn.LayerNorm,
            device=args.device,
            depth = 12,
            num_classes = 7,)
    elif args.model_name == 'ClsCrossPHG':
        model = CrossPHGNet(fusion_type = 'cls_only',
                            alpha = args.alpha,
                            device=args.device,
                            fuse_freq=args.fuse_freq)
    else:
        model = AllAttnPHGNet(device=args.device,
                              alpha=args.alpha,
                              fusion_type = 'cross_attn')
    
    model.apply(init_weights)

    return model



@torch.no_grad()
def evaluate(args,data_loader, model, device):
    model.eval()

    epoch_loss = [] 
    epoch_acc = []
    alpha = model.alpha
    print('alpha = ',alpha)
    for img, labels, pd, pl in tqdm(data_loader):
        # Input:
        # imge: N x 3 x W x H 
        # target: N x num_classes
        img = img.to(device)
        labels = labels.to(device)

        pd = pd.to(device)

        img_pred,topo_pred = model(img,pd)
        

        criterion = torch.nn.CrossEntropyLoss()
        class_label = torch.argmax(labels, dim=1)
        pred = img_pred + alpha*topo_pred
        #acc1 = accuracy(pred, class_label, topk=(1,))[0]
        loss = criterion(pred, class_label)
        acc1 = compute_accuracy(pred,class_label)


        epoch_loss.append(loss.cpu().float().numpy())
        epoch_acc.append(acc1)

    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    print(f'* Acc@1 {epoch_acc}, loss {epoch_loss}')

    return epoch_loss,epoch_acc


def main(args):
    # Meta Info
    print("Start Running: ")
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    input_size = args.image_size
    mean_std = {
        "Prostate": {"mean": [0.6576, 0.4719, 0.6153], "std": [0.2117, 0.2266, 0.1821]},
        "Glaucoma": {"mean": [0.7266, 0.3759, 0.0983], "std": [0.1806, 0.1580, 0.0861]},
        "CBIS_DDSM": {
            "all": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            # "calc": {"mean": [0.6941, 0.6941, 0.6941], "std": [0.1769, 0.1769, 0.1769]},
            "calc": {"mean": [0.5], "std": [0.5]},
            "mass": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        "ISIC": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],"num_classes":7}
    }

    mean = mean_std[args.dataset_name]['mean']
    std = mean_std[args.dataset_name]['std']
    num_classes = mean_std[args.dataset_name]['num_classes']


    # Keep transform from PHG-Net 
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ToPILImage(),
        # transforms.Resize(re_size),
        transforms.Normalize(mean=mean, std=std),
        # transforms.Resize(re_size),
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                scale=[0.7, 1.3]),
        # transforms.RandomCrop(input_size),

    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ToPILImage(),
        transforms.Normalize(mean, std),
        transforms.Resize((input_size, input_size), antialias=True),
        # transforms.ToTensor(),
    ])

    home_dir = args.data_dir

    dataset_paths = {'Prostate': os.path.join(home_dir, 'Prostate/rois'),
                     'Glaucoma': os.path.join(home_dir, 'Glaucoma/Glaucoma'),
                     'CBIS_DDSM': os.path.join(home_dir, 'CBIS_DDSM'),
                     "ISIC": os.path.join(home_dir, "ISIC2018")}

    data_path = dataset_paths[args.dataset_name]

    datasets = {
        # "Prostate": ProstateDataset,
        # "Glaucoma": GlaucomaDataset,
        # "CBIS_DDSM": MammographyDataset,
        "ISIC": ISICDataset
    }

   

    train_set = datasets[args.dataset_name](
        data_dir=data_path,
        transform=train_transform,
        is_train=True,
        load_pd=True,
    )
    # from torchvision.models.swin_transformer.S
    val_set = datasets[args.dataset_name](
        data_dir=data_path,
        transform=val_transform, 
        is_train=False,
        load_pd=True,
    )

    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=12)
    
    val_loader = DataLoader(val_set, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=12)


    # Load Model
    model = load_model(args)    

    model.to(device)
    print("Model = %s" % str(model))

    total_num = sum(p.numel() for p in model.parameters())
    train_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total of Parameters: {}M".format(round(total_num / 1e6, 2)))
    print("Train Parameters: {}M".format(round(train_num/1e6, 2)))

    print("lr: %.3e" % args.lr)

    optimizer =  torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.997)
    
    print(optimizer)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    train_loss_record = []
    train_acc_record = []

    eval_acc_record = []
    eval_loss_record = []

    alpha = model.alpha
    print('Training alpha is ',alpha)
    for epoch in tqdm(range(args.epochs)):
        epoch_loss = [] 
        epoch_acc = []

        model.train()
        for img, labels, pd, pl in tqdm(train_loader):
            # Input:
                # imge: N x 3 x W x H 
                # target: N x num_classes
            optimizer.zero_grad()
            img = img.to(device)

            labels = labels.to(device) # N x num_class
            pd = pd.to(device)

            img_pred,topo_pred = model(img,pd)

            criterion = torch.nn.CrossEntropyLoss()
            class_label = torch.argmax(labels, dim=1)
            
            #acc1 = accuracy(pred, class_label, topk=(1,))[0]
            pred = img_pred + alpha*topo_pred
            acc1 = compute_accuracy(pred,class_label)
            # Back Prop. #################################################################
            loss = criterion(pred,class_label)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss.append(loss.detach().cpu().float().numpy())
            epoch_acc.append(acc1)
            # Update record
            train_loss_record.append(loss.detach().cpu().float().numpy())
            train_acc_record.append(acc1)

        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        print(f'* Acc@1 {epoch_acc}, loss {epoch_loss}')


        # Evaluate #########################################################################    
        loss_eval,acc_eval = evaluate(args,val_loader, model, device)
        eval_loss_record.append(loss_eval)
        eval_acc_record.append(acc_eval)
        print(f"Accuracy of the network on the test images: {acc_eval}")
        max_accuracy = max(max_accuracy, acc_eval)
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if args.output_dir and (epoch % 10 == 1 or epoch + 1 == args.epochs):
            check_pt_path = os.path.join(args.output_dir,'crossPHG_ckpt__{}_{}.pth'.format(args.remark,str(epoch)))
            to_save = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_loss':train_loss_record,
                        'eval_loss':eval_loss_record,
                        'train_acc':train_acc_record,
                        'eval_acc':eval_acc_record}
    
            torch.save(to_save,check_pt_path)

    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implementation of SwinV2 pretrain on Imagenet",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--epochs', default=20, type=int)
    
    # dataset parameter
    parser.add_argument('--batch_size',default='32',type=int,
                        help='batchsize of dataloaders')
    parser.add_argument('--dataset_name',default='ISIC',type=str,
                        help='dataset name')
    parser.add_argument('--data_dir',default='data/raw_data')
    parser.add_argument('--image_size',default=224,type=int)
    
    # optimize parameter
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--lr', type=float, default=5e-2,
                        help='learning rate (absolute lr)')
    parser.add_argument('--seed', default=42, type=int)

    # output
    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--remark', default='remark',
                        help='Model Remark')
    
    
    # model
    parser.add_argument('--model_name', default='SwinV2B20', type=str,
                        help='Name of model to train')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='alpha (control topo feats)')
    parser.add_argument('--fuse_freq', default=1, type=int,
                        help='fuse frequency, default fuse in every layer')
    
    

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)


# python -m crossPHG_main --batch_size 4 --device cpu --lr 1e-3 --epochs 50 --model_name crossPHG


# python -m crossPHG_main --batch_size 64 --device cuda --lr 1e-2 --epochs 50 --model_name crossPHG --alpha 0.1 --remark topoLoss0.1
# python -m crossPHG_main --batch_size 64 --device cuda --lr 1e-2 --epochs 50 --model_name attnPHG --alpha 0.1 --remark allAttn0.1
# python -m crossPHG_main --batch_size 64 --device cuda --lr 1e-3 --epochs 50 --model_name crossPHG --alpha 0 --remark topoloss0

# python -m crossPHG_main --batch_size 64 --device cuda --lr 1e-2 --epochs 50 --model_name ClsCrossPHG --alpha 0.1 --remark topoloss0.1_2


# python -m crossPHG_main --batch_size 64 --device cuda --lr 5e-3 --epochs 50 --model_name crossPHG --fuse_freq 1 --alpha 0.2 --remark topoloss50_a0.2f1 