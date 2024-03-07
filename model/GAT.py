import torch
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GCNConv(in_channels,out_channels)#GATConv(in_channels, out_channels, heads=8, concat=False)

    def forward(self, batch_data):
        x, edge_index = batch_data.x, batch_data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        return x

if __name__ == '__main__':
    pass

