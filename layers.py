"""
GCNH Layer
"""

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch_scatter import scatter


class GCNH_layer(Module):
    def __init__(self, nfeat, nhid, maxpool):
        super(GCNH_layer, self).__init__()
        
        self.nhid = nhid
        self.maxpool = maxpool
        
        # Two MLPs, one to encode center-node embedding,
        # the other for the neighborhood embedding
        self.MLPfeat = nn.Sequential(
            nn.Linear(nfeat, self.nhid),
            nn.LeakyReLU()
        )
        self.init_weights(self.MLPfeat)
        
        self.MLPmsg = nn.Sequential(
            nn.Linear(nfeat, self.nhid),
            nn.LeakyReLU()
        )
        self.init_weights(self.MLPmsg)
        
        # Parameter beta
        self.beta = nn.Parameter(0.0 * torch.ones(size=(1, 1)), requires_grad=True) 
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
            
    def forward(self, feat, adj, cur_idx=None,row=None,col=None):
        """
        feat: feature matrix 
        adj: adjacency matrix
        cur_idx: index of nodes in current batch
        row, col: used for maxpool aggregation
        """
        if cur_idx == None:
            cur_idx = range(feat.shape[0])
        
        # Transform center-node and neighborhood messages
        h = self.MLPfeat(feat)
        z = self.MLPmsg(feat)
        
        # Aggregate messages
        beta = torch.sigmoid(self.beta)
        
        if not self.maxpool: # sum or mean
            hp = beta * z + (1-beta) * torch.matmul(adj, h)
        else:
            hh = torch.zeros(adj.shape[0], self.nhid).cuda()
            _ = scatter(h[row], col, dim=0, out=hh, reduce="max")
            hp = beta * z + (1 - beta) * hh
        
        return hp, beta
