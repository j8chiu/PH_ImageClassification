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
    def __init__(self, data_dir='data/raw_data/ISIC2018', transform=None, is_train=True):
        """SKin Lesion"""
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train

        # self.pd_dir = join(self.img_path, "persistent_diagram_1_old")
        # self.pl_dir = join(self.img_path, "persistent_landscape_old")
        # self.data, self.targets = self.get_data(fold)

        self.classes_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.classes = list(range(len(self.classes_name)))

        if self.is_train:
            self.data_path = os.path.join(self.data_dir,'train')
            csv = os.path.join(self.data_dir,'label_train.csv')
        else:
            self.data_path = os.path.join(self.data_dir,'test')
            csv = os.path.join(self.data_dir,'label_test.csv')

        self.data = os.listdir(self.data_path)

        csvfile = pd.read_csv(csv)
        raw_data = csvfile.values # 

        self.data_names = []
        self.targets = []
        for data_list in raw_data:
            image_name = data_list[0]+'.jpg'
            self.data_names .append(os.path.join(self.data_path,image_name))
            label = data_list[1:] # one_hot label
            self.targets.append(label)

        # self.target_img_dict = {}
        # targets = np.array(self.targets)
        # for target in self.classes:
        #     indexes = np.nonzero(targets == target)[0]
        #     self.target_img_dict.update({target: indexes})

    def __getitem__(self, i):
        """
                Args:
                    index (int): Index
                Returns:
                    tuple: (sample, target) where target is class_index of the
                           target class.
                """
        path = self.data_names[i]
        #case = os.path.basename(path).split(".")[0]
        target = torch.tensor(self.targets[i].astype(np.int64)) # one-hot

        img = io.imread(path)
        # img = pil_loader(path)

        pd = []
        pl = []
        # for mode in ['r', 'g', 'b', 'gray', 'r_inverse', 'g_inverse', 'b_inverse', 'gray_inverse']:
        #     for d in [0, 1]:
        #         pd.append(np.load(join(self.pd_dir, case, f'{mode}_{d}.npy')))
        #         pl.append(np.load(join(self.pl_dir, case, f'{mode}_{d}.npy')))

        # for d in [0, 1]:
        #     pd.append(np.load(join(self.pd_dir, case, f'dim_{d}.npy')))
        #     pl.append(np.load(join(self.pl_dir, case, f'dim_{d}.npy')))

        # pd = [np.load(join(self.pd_dir, case, f'dim_{x}.npy')) for x in range(0, 2)]
        # pd = process_pd(pd, dims=[0, 1], samples=73, case=case)

        #pd = process_pd(pd, dims=[0, 1], samples=85, case=case)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, #pd, np.vstack(pl)

        # return img, target, pd, np.vstack(pl), \
        #     cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), path

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    train_transform = transforms.Compose([
        transforms.ToTensor(),])

    data = ISICDataset(transform=train_transform)
    dataloader = DataLoader(data, batch_size=2)
    for item in dataloader:
        print(item)
