import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.pd_encoder import CombinedModel, PersistenceDiagramEncoder, Classifier
from dataset.isicloader import ISICDataset
import numpy as np

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

def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total

def test_model(model, device, dataloader):
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            pd, labels = batch
            pd = pd.to(device)
            labels = labels.argmax(dim=1) if labels.ndim > 1 else labels
            labels = labels.to(device)

            outputs = model(pd)
            accuracy = compute_accuracy(outputs, labels)
            total_accuracy += accuracy
            print(f'Batch {batch_idx+1}, Batch Acc: {accuracy:.2f}%')

    average_accuracy = total_accuracy / len(dataloader)
    print(f'Average Test Accuracy: {average_accuracy:.2f}%')

def load_model(model_path, device):
    pd_encoder = PersistenceDiagramEncoder(input_dim=4)
    classifier = Classifier(input_dim=1024, num_classes=7)
    model = CombinedModel(pd_encoder, classifier).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'path_to_your_model_checkpoint.pth.tar' # change this, you motherfkers
    model = load_model(model_path, device)

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = ISICDataset(transform=transform, is_train=False)  
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    test_model(model, device, test_dataloader)
