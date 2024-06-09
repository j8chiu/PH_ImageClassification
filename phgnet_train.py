import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.pd_encoder import CombinedModel, PersistenceDiagramEncoder, Classifier
from dataset.isicloaderpthnet import ISICDataset  
import numpy as np
import os
from modifies_resnet50 import ResNet50

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda")

#hyperparameters
epochs = 50
vision_topo_ratio = 0.1
learning_rate = 0.005

#data distribution for ISIC, used for data transformation
mean_std = {
    "ISIC": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
}
mean = mean_std['ISIC']['mean']
std = mean_std['ISIC']['std']


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

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total

def save_checkpoint(state, filename="checkpoint.pth.tar", checkpoint_list=[]):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

    # Maintain only the last three checkpoints
    checkpoint_list.append(filename)
    if len(checkpoint_list) > 3:
        # Remove the oldest checkpoint file
        oldest_checkpoint = checkpoint_list.pop(0)
        if os.path.exists(oldest_checkpoint):
            os.remove(oldest_checkpoint)
            print(f"Removed oldest checkpoint: {oldest_checkpoint}")

    return checkpoint_list

def train_model():

    #transform images
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                scale=[0.7, 1.3]),
    ])

    #transform 

    dataset = ISICDataset(transform=train_transform, is_train=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn,num_workers=4)

    model = ResNet50().to(device)

    criterion_1 = nn.CrossEntropyLoss().cuda()
    criterion_2 = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    checkpoint_list = []  # List to track checkpoint filenames

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            pd, labels, imgs = batch
            pd = pd.to(device)
            labels = labels.argmax(dim=1) if labels.ndim > 1 else labels
            labels = labels.to(device)
            imgs = imgs.to(device)

            cv_outputs, topo_outputs = model(imgs, pd)
            optimizer.zero_grad()
            loss1 = criterion_1(cv_outputs, labels)
            loss2 = criterion_2(topo_outputs, labels)
            loss = loss1 + vision_topo_ratio*loss2
            if torch.isnan(loss):
                print("NaN detected in loss at epoch", epoch, "batch index", batch_idx)
                continue

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Compute and log batch accuracy
            accuracy = compute_accuracy(cv_outputs + vision_topo_ratio*topo_outputs, labels)
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Batch Acc: {accuracy:.2f}%')

        # Save checkpoint at the end of each epoch
        checkpoint_filename = f"phg_net_checkpoint_epoch_{epoch+1}.pth.tar"
        checkpoint_list = save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_filename, checkpoint_list=checkpoint_list)

    print("Training complete.")

if __name__ == "__main__":
    train_model()