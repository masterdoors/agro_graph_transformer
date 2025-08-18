import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim , input_dim , bias=True, dtype=torch.double) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim , output_dim , bias=True,dtype=torch.double ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.tn = nn.Tanh()
        self.rel = nn.ReLU()
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return self.rel(self.tn(y))