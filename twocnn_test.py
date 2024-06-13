import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.pd_encoder import CombinedModel, PersistenceDiagramEncoder, Classifier
from dataset.isicloaderpthnet import ISICDataset
import numpy as np
import os
from two_cnn import ResNet50

device = torch.device("cuda")
vision_topo_ratio = 0.1
learning_rate = 0.001
pure_cnn_ratio = 0.4
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
    pds, labels, img = zip(*[(remove_inf(item[2]), item[1], item[0]) for item in batch if item[2].size > 0])
    if len(pds) == 0:
        return None
    max_length = max(pd.shape[0] for pd in pds)
    pds_padded = pad_persistence_diagrams(pds, max_length)
    labels = torch.stack(labels)
    pds_padded = torch.tensor(pds_padded, dtype=torch.float32)
    images = torch.stack(img)
    return pds_padded, labels, images

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total

def test_model():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize((224, 224), antialias=True),
    ])

    test_dataset = ISICDataset(transform=test_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = ResNet50().to(device)
    model.load_state_dict(torch.load('twocnn_checkpoint_epoch_99.pth.tar')['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()

    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        total_samples = 0

        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                continue
            pd, labels, imgs = batch
            pd = pd.to(device)
            labels = labels.argmax(dim=1) if labels.ndim > 1 else labels
            labels = labels.to(device)
            imgs = imgs.to(device)

            cv_outputs, topo_outputs,res_outputs = model(imgs, pd)
            loss = criterion(cv_outputs + vision_topo_ratio*topo_outputs + pure_cnn_ratio*res_outputs, labels)
            total_loss += loss.item() * labels.size(0)
            accuracy = compute_accuracy(cv_outputs+ 2.0*topo_outputs + 0.1*res_outputs, labels)
            total_accuracy += accuracy * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_accuracy / total_samples
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.2f}%")

if __name__ == "__main__":
    test_model()