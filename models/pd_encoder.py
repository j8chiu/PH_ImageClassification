import torch
import torch.nn as nn
import torch.nn.functional as F



class PersistenceDiagramEncoder(nn.Module):
    def __init__(self, input_dim):
        super(PersistenceDiagramEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4)  
        self.fc2 = nn.Linear(4, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 1024)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))


        x = x.transpose(1, 2)


        x = F.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2)

        return x


class Classifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=7):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class CombinedModel(nn.Module):
    def __init__(self, pd_encoder, classifier):
        super(CombinedModel, self).__init__()
        self.pd_encoder = pd_encoder
        self.classifier = classifier

    def forward(self, pd):
        topological_features = self.pd_encoder(pd)
        output = self.classifier(topological_features)
        return output
