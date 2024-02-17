import torch 
import torch.nn as nn
import torch.nn.functional as F

class MeaningExtraction(nn.Module): 
    def __init__(self, input_size, embedding_size):
        super(MeaningExtraction, self).__init__()
        self.embedding = nn.EMbedding(input_size, embedding_size)

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded
    
class SideNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 
        super(SideNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    