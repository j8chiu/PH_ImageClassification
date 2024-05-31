import os.path
import numpy as np
from skimage import io
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from einops import repeat
from torchvision import transforms
import torch

class ISICDataset(Dataset):
    def __init__(self, data_dir='D:\\UCSD\\2024 spring\\214\\project\\chiupart\\PH_ImageClassification\\data\\ISIC', transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train

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
        
        pd = data['diagram'][0]  # Assuming this returns a list of (birth, death, group_index)
        pl = data['landscape']
        label_values = list(data['label'].values())
        label = torch.tensor(label_values, dtype=torch.float32)

        
        return img, label, pd, pl

    def __len__(self):
        return len(self.npy_files)


#can be removed afterwards, this is just for me to run the dataloader without any error
def pad_persistence_diagrams(pds, max_length):
    padded_pds = []
    for pd in pds:
        pad_size = max_length - pd.shape[0]
        padded_pd = np.pad(pd, ((0, pad_size), (0, 0)), 'constant', constant_values=0)
        padded_pds.append(padded_pd)
    return np.stack(padded_pds)


def remove_inf(data):
    data[np.isinf(data)] = np.nan 
    mean_val = np.nanmean(data)  
    data[np.isnan(data)] = mean_val  
    return data

def collate_fn(batch):
    pds, labels, img = zip(*[(remove_inf(item[2]), item[1],item[0]) for item in batch if item[2].size > 0])
    if len(pds) == 0:
        return None
    max_length = max(pd.shape[0] for pd in pds)
    pds_padded = pad_persistence_diagrams(pds, max_length)
    labels = torch.stack(labels)
    pds_padded = torch.tensor(pds_padded, dtype=torch.float32)
    images = torch.stack(img)
    return pds_padded, labels, images


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ISICDataset(transform=transform, is_train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    a, b, c = next(iter(dataloader))
    print(a.shape)