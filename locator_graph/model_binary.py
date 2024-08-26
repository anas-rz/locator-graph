import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, 2)  # Binary classification (2 classes)

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Final dense layer for classification
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
