import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.pd_encoder import CombinedModel, PersistenceDiagramEncoder, Classifier
from dataset.isicloader import ISICDataset  
import numpy as np
import os


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
    pds, labels = zip(*[(remove_inf(item[2]), item[1]) for item in batch if item[2].size > 0])
    if len(pds) == 0:
        return None
    max_length = max(pd.shape[0] for pd in pds)
    pds_padded = pad_persistence_diagrams(pds, max_length)
    labels = torch.stack(labels)
    pds_padded = torch.tensor(pds_padded, dtype=torch.float32)
    return pds_padded, labels


def init_weights(m):
    if type(m) == nn.Linear:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ISICDataset(transform=transform, is_train=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    pd_encoder = PersistenceDiagramEncoder(input_dim=4)
    classifier = Classifier(input_dim=1024, num_classes=7)
    model = CombinedModel(pd_encoder, classifier).to(device)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    num_epochs = 10
    checkpoint_list = []  # List to track checkpoint filenames

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            pd, labels = batch
            pd = pd.to(device)
            labels = labels.argmax(dim=1) if labels.ndim > 1 else labels
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pd)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print("NaN detected in loss at epoch", epoch, "batch index", batch_idx)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Compute and log batch accuracy
            accuracy = compute_accuracy(outputs, labels)
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Batch Acc: {accuracy:.2f}%')

        # Save checkpoint at the end of each epoch
        checkpoint_filename = f"checkpoint_epoch_{epoch+1}.pth.tar"
        checkpoint_list = save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_filename, checkpoint_list=checkpoint_list)

    print("Training complete.")

if __name__ == "__main__":
    train_model()