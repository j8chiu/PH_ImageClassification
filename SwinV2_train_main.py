# stdlib
import argparse
from argparse import RawTextHelpFormatter
from tqdm import tqdm
import os
from pathlib import Path

from torchvision import transforms
import torch
import torch.nn as nn
from timm.utils import accuracy
import numpy as np
import time
import datetime


# dataset
from dataset.isic_dataset import ISICDataset
# from dataset.prostate_dataset import ProstateDataset
# from dataset.glaucoma_dataset import GlaucomaDataset
# from dataset.mammography_dataset import MammographyDataset
from torch.utils.data.dataloader import DataLoader

# model
from models.swin_transformer_v2 import swin_v2_b
from torch.nn.utils import clip_grad_norm_

from pytorch_pretrained_vit import ViT

@torch.no_grad()
def evaluate(args,data_loader, model, device):
    # switch to evaluation mode
    model.eval()

    epoch_loss = [] 
    epoch_acc = []

    for img, target in tqdm(data_loader):
        # Input:
            # imge: N x 3 x W x H 
            # target: N x num_classes
        img = img
        pred,_ = model(img) #N x num_classes
    
        # output is a list, each element in a list is a tensor contains class probability.
        loss = 0
        acc1 = 0
        target = target

        criterion = torch.nn.CrossEntropyLoss()
        loss += criterion(pred, target)
        class_label = torch.argmax(target, dim=1)
        acc1 += accuracy(pred, class_label, topk=(1,))[0]

        epoch_loss.append(loss.cpu().float().numpy())
        epoch_acc.append(acc1.cpu().float().numpy())

    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    print(f'* Acc@1 {epoch_acc}, loss {epoch_loss}')

    return epoch_loss,epoch_acc


def main(args):
    # Meta Info
    device = torch.device(args.device)
    input_size = 224
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

    data_loading_start_time = time.time()

    train_set = datasets[args.dataset_name](
        data_dir=data_path,
        transform=train_transform, is_train=True,
        device=device
    )
    # from torchvision.models.swin_transformer.S
    val_set = datasets[args.dataset_name](
        data_dir=data_path,
        transform=val_transform, is_train=False,
        device=device
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False)
    data_loading_end_time = time.time()
    data_loading_time = data_loading_end_time - data_loading_start_time
    print(f"Data loading time: {datetime.timedelta(seconds=int(data_loading_time))}")

    # Load Model
    # Load Model
    model = swin_v2_b(weights="IMAGENET1K_V1", num_classes=1000,).to(device)
    model.head = nn.Linear(in_features=1024, out_features=num_classes, bias=True).to(device)

    # model = ViT('B_16_imagenet1k', pretrained=True,image_size=224).to(device)
    # model.fc = nn.Linear(in_features=768, out_features=num_classes, bias=True).to(device)

    # Freeze model but head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)
    print("Model = %s" % str(model))

    total_num = sum(p.numel() for p in model.parameters())
    train_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total of Parameters: {}M".format(round(total_num / 1e6, 2)))
    print("Train Parameters: {}".format(round(train_num, 2)))

    print("lr: %.3e" % args.lr)

    optimizer =  torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        )
    
    print(optimizer)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    train_loss_record = []
    train_acc_record = []

    eval_acc_record = []
    eval_loss_record = []

    for epoch in tqdm(range(args.epochs)):
        epoch_loss = [] 
        epoch_acc = []

        for img, target in tqdm(train_loader):
            # Input:
                # imge: N x 3 x W x H 
                # target: N x num_classes
            img = img

            model_inference_start = time.time()
            pred,_ = model(img) #N x num_classes
            model_inference_end = time.time()
            model_loading_time = model_inference_end - model_inference_start
            print(f"Model inferencing time: {datetime.timedelta(seconds=int(model_loading_time))}")

            # output is a list, each element in a list is a tensor contains class probability.
            loss = 0
            acc1 = 0

            criterion = torch.nn.CrossEntropyLoss()
            target = target
            loss += criterion(pred, target)
            class_label = torch.argmax(target, dim=1)
            acc1 += accuracy(pred, class_label, topk=(1,))[0]

        # Back Prop. #################################################################
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        for p in model.parameters(): # addressing gradient vanishing
            if p.requires_grad and p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0)
        clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        epoch_loss.append(loss.detach().cpu().float().numpy())
        epoch_acc.append(acc1.detach().cpu().float().numpy())
        # Update record
        train_loss_record.append(loss.detach().cpu().float().numpy())
        train_acc_record.append(loss.detach().cpu().float().numpy())

        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        print(f'* Acc@1 {epoch_acc}, loss {epoch_loss}')


        # Evaluate #########################################################################    
        loss,acc1 = evaluate(args,val_loader, model, device)
        eval_loss_record.append(loss)
        eval_acc_record.append(acc1)
        print(f"Accuracy of the network on the test images: {acc1}")
        max_accuracy = max(max_accuracy, acc1)
        print(f'Max accuracy: {max_accuracy:.2f}%')

    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.output_dir:
        check_pt_path = os.path.join(args.output_dir,'linearProb_{}_checkpoint_{}_{}.pth'.format(args.dataset_name,args.model_name,str(epoch)))
        to_save = {'model': model.state_dict(),
                   'train_loss':train_loss_record,
                   'eval_loss':eval_loss_record,
                   'train_acc':train_acc_record,
                   'eval_acc':eval_acc_record}
        
        torch.save(to_save,check_pt_path)




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
    
    # optimize parameter
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--lr', type=float, default=5e-2,
                        help='learning rate (absolute lr)')
    parser.add_argument('--seed', default=42, type=int)

    # output
    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--model_name', default='SwinV2B20', type=str,
                        help='Name of model to train')
    

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)


# python -m SwinV2_train_main --batch_size 64 --device cuda --lr 1e-2
# python -m SwinV2_train_main --batch_size 32 --device cuda
