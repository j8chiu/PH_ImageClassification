import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Assuming the ISICDataset class is imported from isic_dataset.py
from dataset.isicloader import ISICDataset


def test_dataset(data_dir, is_train=False):
    """Function to test the dataset by loading and displaying some entries."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = ISICDataset(data_dir=data_dir, transform=transform, is_train=is_train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (img, label, pd, pl) in enumerate(dataloader):
        print("Image Shape:", img.shape)
        print("Label:", label[0])
        print("Persistence Diagram:", pd[0])
        print("Persistence Landscape:", pl[0])


if __name__ == "__main__":
    data_dir = '/Users/chiuchiu/Downloads/ISIC2018'
    test_dataset(data_dir, is_train=False)
