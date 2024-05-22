# stdlib
import argparse
from argparse import RawTextHelpFormatter
from tqdm import tqdm
import os

from torchvision import transforms

# dataset
from dataset.isic_dataset import ISICDataset
from dataset.prostate_dataset import ProstateDataset
from dataset.glaucoma_dataset import GlaucomaDataset
from dataset.mammography_dataset import MammographyDataset

def main(args):
    # Meta Info
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
        "ISIC": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }

    mean = mean_std[args.dataset_name]['mean']
    std = mean_std[args.dataset_name]['std']


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
    label_files = {
        'Prostate': os.path.join(home_dir, "Prostate/rois-with-grade.csv"),
        'Glaucoma': os.path.join(home_dir, "Glaucoma/label.csv"),
        'CBIS_DDSM': {
            'all': os.path.join(home_dir, "CBIS_DDSM/all_cases.csv"),
            'calc': os.path.join(home_dir, "CBIS_DDSM/calc.csv"),
            'mass': os.path.join(home_dir, "CBIS_DDSM/mass.csv"),
        },
        "ISIC": os.path.join(home_dir, "ISIC/label.csv")
    }

    data_path = dataset_paths[args.dataset_name]
    label_file = label_files[args.dataset_name]

    datasets = {
        "Prostate": ProstateDataset,
        "Glaucoma": GlaucomaDataset,
        "CBIS_DDSM": MammographyDataset,
        "ISIC": ISICDataset
    }

    train_set = datasets[args.dataset_name](
        img_path=data_path,
        label_file=label_file,
        transform=train_transform, fold=fold, train=True
    )
    # from torchvision.models.swin_transformer.S
    val_set = datasets[data_base](
        img_path=data_path,
        label_file=label_file,
        transform=val_transform, fold=fold, train=False
    )


    



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implementation of SwinV2 pretrain on Imagenet",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_name',default='ISIC',type=str,
                        help='dataset name')
    parser.add_argument('--data_dir',default='data/raw_data')
    

    args = parser.parse_args()
    main(args)
