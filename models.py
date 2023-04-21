"""
Define GCNH model
"""

import torch.nn as nn
import torch.nn.functional as F
from layers import GCNH_layer
import torch
from utils import *

class GCNH(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout, nlayers, maxpool):
        super(GCNH, self).__init__()
        
        self.nhid = nhid
        self.dropout = dropout
        self.nlayers = nlayers

        # Define layers
        layer_sizes = [nfeat] + [nhid] * (self.nlayers - 1)
        self.layers = nn.ModuleList([GCNH_layer(layer_sizes[i], nhid, maxpool) for i in range(self.nlayers)])

        # MLP for classification
        self.MLPcls = nn.Sequential(
            nn.Linear(self.nhid, nclass),
            nn.LogSoftmax(dim=1)
        )
        self.init_weights(self.MLPcls)
        
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
            
    def forward(self, feat, adj, cur_idx=None, verbose=False, row=None, col=None):
        """
        feat: feature matrix 
        adj: adjacency matrix
        cur_idx: index of nodes in current batch
        row, col: used for maxpool aggregation
        """
        if cur_idx == None:
            cur_idx = range(feat.shape[0])

        hp = feat
        for i in range(self.nlayers):
            hp, beta = self.layers[i](hp, adj, cur_idx=cur_idx,row=row,col=col)
            if verbose:
                print("Layer: ", i, " beta: ", beta.item())
            hp = F.dropout(hp, self.dropout, training=self.training)

        return self.MLPcls(hp[cur_idx])
        
