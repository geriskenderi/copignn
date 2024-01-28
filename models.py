import torch
import lightning as L
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GraphConv
from utils import maxcut_hamiltonian, mis_hamiltonian

MODEL_KEY_DICT = {
    0: 'GCN',
    1: 'GAT',
    2: 'GATv2',
    3: 'GraphConv'
}

class COPIGNN(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, co_problem, lr=1e-3, out_dim=1, num_heads=1, layer_type=0):
        """
        The base class of a gnn classifier. The layer type indicates which gnn to use.
        The different options are:
        0 - GCN
        1 - GAT
        2 - GATv2
        """
        super(COPIGNN, self).__init__()
        assert layer_type in [0,1,2,3], "Unsuported GNN MP layer!"
        assert co_problem in ['maxcut', 'mis'], "Unsuported CO Problem!"

        self.layer_type = layer_type
        self.lr = lr
        self.loss_fn = maxcut_hamiltonian if co_problem == 'maxcut' else mis_hamiltonian
        
        if layer_type == 0:
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif layer_type == 1:
            self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads)
            self.conv2 = GATConv(hidden_dim*num_heads, out_dim, heads=num_heads, concat=False)
        elif layer_type == 2:
            self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=num_heads)
            self.conv2 = GATv2Conv(hidden_dim*num_heads, out_dim, heads=num_heads, concat=False)
        elif layer_type == 3:
            self.conv1 = GraphConv(in_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, out_dim) 
        self.save_hyperparameters()


    def forward(self, x, edge_index):        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        logits = self.conv2(x, edge_index)

        return torch.sigmoid(logits)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        pred = self.forward(x, edge_index)
        loss = self.loss_fn(edge_index, pred)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        pred = self.forward(x, edge_index)
        loss = self.loss_fn(edge_index, pred.round()) # round to nearest int for validatio, i.e. calc QUBO
        self.log('val_loss', loss, prog_bar=True)