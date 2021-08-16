<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool
from torch_geometric.utils import dropout_adj


# GCN based model
class DTA_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(DTA_GCN, self).__init__()

        print('DTA_GCN Loading ...')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index, target_weight)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


# GAT based model
class DTA_GAT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(DTA_GAT, self).__init__()

        print('DTA_GAT Loading ...')
        self.n_output = n_output
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=2, dropout=dropout)
        self.mol_conv2 = GATConv(num_features_mol * 2, num_features_mol * 2, heads=2, dropout=dropout)
        self.mol_conv3 = GATConv(num_features_mol * 4, num_features_mol * 4, dropout=dropout)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GATConv(num_features_pro, num_features_pro * 2, heads=2, dropout=dropout)
        self.pro_conv3 = GATConv(num_features_pro * 4, num_features_pro * 4, dropout=dropout)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


# GCN based model
class CPI_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(CPI_GCN, self).__init__()

        print('CPI_GCN loading ...')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index, target_weight)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        return out


# GAT based model
class CPI_GAT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(CPI_GAT, self).__init__()

        print('CPI_GAT loading ...')
        self.n_output = n_output
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=2, dropout=dropout)
        self.mol_conv2 = GATConv(num_features_mol * 2, num_features_mol * 2, heads=2, dropout=dropout)
        self.mol_conv3 = GATConv(num_features_mol * 4, num_features_mol * 4, dropout=dropout)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GATConv(num_features_pro, num_features_pro * 2, heads=2, dropout=dropout)
        self.pro_conv3 = GATConv(num_features_pro * 4, num_features_pro * 4, dropout=dropout)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        return out
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool
from torch_geometric.utils import dropout_adj


# GCN based model
class DTA_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(DTA_GCN, self).__init__()

        print('DTA_GCN Loading ...')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index, target_weight)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


# GAT based model
class DTA_GAT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(DTA_GAT, self).__init__()

        print('DTA_GAT Loading ...')
        self.n_output = n_output
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=2, dropout=dropout)
        self.mol_conv2 = GATConv(num_features_mol * 2, num_features_mol * 2, heads=2, dropout=dropout)
        self.mol_conv3 = GATConv(num_features_mol * 4, num_features_mol * 4, dropout=dropout)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GATConv(num_features_pro, num_features_pro * 2, heads=2, dropout=dropout)
        self.pro_conv3 = GATConv(num_features_pro * 4, num_features_pro * 4, dropout=dropout)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


# GCN based model
class CPI_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(CPI_GCN, self).__init__()

        print('CPI_GCN loading ...')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index, target_weight)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        return out


# GAT based model
class CPI_GAT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=33, num_features_mol=78, output_dim=128, hidden_channels=64,
                 dropout=0.2):
        super(CPI_GAT, self).__init__()

        print('CPI_GAT loading ...')
        self.n_output = n_output
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=2, dropout=dropout)
        self.mol_conv2 = GATConv(num_features_mol * 2, num_features_mol * 2, heads=2, dropout=dropout)
        self.mol_conv3 = GATConv(num_features_mol * 4, num_features_mol * 4, dropout=dropout)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GATConv(num_features_pro, num_features_pro * 2, heads=2, dropout=dropout)
        self.pro_conv3 = GATConv(num_features_pro * 4, num_features_pro * 4, dropout=dropout)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_mol, data_pro):
        # get graph input
        # mol_x, mol_edge_index, mol_edge_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        mol_x, mol_edge_index, mol_weight, mol_batch = data_mol.x, data_mol.edge_index, data_mol.edge_weight, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_weight, target_batch = data_pro.x, data_pro.edge_index, data_pro.edge_weight, data_pro.batch
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index.size(), 'batch', target_batch.size())

        # x = self.mol_conv1(mol_x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        # x = self.mol_conv2(x, mol_edge_index, mol_edge_weight)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # xt = self.pro_conv1(target_x, target_edge_index, target_weight)
        # print(target_x.size(),target_edge_index.size())
        # print(target_x)
        # print(target_edge_index)
        xt = self.pro_conv1(target_x, target_edge_index, target_weight)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        # xt = self.pro_conv2(xt, target_edge_index, target_weight)
        xt = self.pro_conv2(xt, target_edge_index)

        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))
        return out
>>>>>>> 496001edb823488862f0910433051342919e21f0
