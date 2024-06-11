import os.path

#from dataset.pd_utils import *
import numpy as np
#from batchgenerators.utilities.file_and_folder_operations import *
from skimage import io
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from einops import repeat
import cv2
from torchvision import transforms
import torch


class ISICDataset(Dataset):
    def __init__(self, data_dir='data/data_raw/ISIC2018', 
                 transform=None, 
                 is_train=True,
                 load_pd = False):
        
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.load_pd = load_pd

        self.classes_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.classes = list(range(len(self.classes_name)))

        self.data_path = os.path.join(self.data_dir, 'train_ph' if self.is_train else 'test_ph')
        self.npy_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.npy')]

    def __getitem__(self, index):
        npy_path = self.npy_files[index]
        data = np.load(npy_path, allow_pickle=True).item()
        img = io.imread(data['image_path'])
        if self.transform:
            img = self.transform(img)
        
        label_values = list(data['label'].values())
        label = torch.tensor(label_values, dtype=torch.float32)

        if self.load_pd:
            pd = data['diagram'][0]  # Assuming this returns a list of (birth, death, group_index)
            pl = data['landscape']
            return img, label, pd, pl
        else:
            return img, label

    def __len__(self):
        return len(self.npy_files)

# Example use
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),  # Example transformation
    ])
    dataset = ISICDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for img, label, pd, pl in dataloader:
        print("Image Shape:", img.shape)
        print("Label:", label)
        print("Persistence Diagram:", pd)
        print("Persistence Landscape:", pl)