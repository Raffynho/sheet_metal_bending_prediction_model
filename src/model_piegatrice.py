import torch
import torch.nn as nn

class BendingModel(nn.Module):
    def __init__(self, num_features, hidden_sizes=(16, 4)):
        super(BendingModel, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        
        in_size = num_features
        for h_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_size, h_size))
            in_size = h_size 
            
        self.output_layer = nn.Linear(in_size, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x)) 
            
        x = self.output_layer(x)  
        return x