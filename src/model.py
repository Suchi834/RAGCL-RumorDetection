import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv

class RAGCL(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(RAGCL, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = torch.relu(x)
        x = self.conv2(g, x)
        return x

