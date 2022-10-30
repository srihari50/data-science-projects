import pandas as pd ###for data manipulation###
import matplotlib.pyplot as plt ###for data visualization###
import dtale as dt ###for auto eda###
import networkx as nx ### for network analysis###
from community import community_louvain ###for louvain algorithm###
import matplotlib.cm as cm
import networkx.algorithms.community as nx_com

###importing the dataset###
routes = pd.read_excel("/content/Routes Data.xlsx")
routes.info()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
routes['DURATION_MIN'] = scaler.fit_transform(routes[['DURATION_MIN']])
routes['DISTANCE_KM'] = scaler.fit_transform(routes[['DISTANCE_KM']])
routes['SPEED'] = scaler.fit_transform(routes[['SPEED']])
routes['DURATION_HRS'] = scaler.fit_transform(routes[['DURATION_HRS']])

###create an edge list from source and destinations###
g = nx.Graph(nx.from_pandas_edgelist(routes, source = 'SOURCE', target = 'DESTINATION', edge_attr = ['DURATION_MIN', 'DURATION_HRS', 'SPEED', 'DISTANCE_KM']))
pos = nx.spring_layout(g)
nx.draw_networkx(g, pos, node_size = 25, node_color = 'blue')
print(nx.info(g))

d = nx.degree_centrality(g)
print(d)

###creating matrix###
mat = nx.to_numpy_matrix(g)

#convert matrix into dataframe
df = pd.DataFrame(mat)

#auto eda on matrix dataset
dt.show(df)

###louvanian algorithm###

partition = community_louvain.best_partition(g, weight = 'weight', resolution = 1)
pos = nx.spring_layout(g, k=100)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

nx.draw_networkx(g, pos, partition.keys(), node_size=100, cmap=cmap, alpha = 0.9, node_color=list(partition.values()))

community_louvain.modularity(partition, g, weight = 'weight') ###modularity###
nx_com.partition_quality(g, nx_com.label_propagation_communities(g)) ###partition qality###
nx_com.coverage(g, nx_com.label_propagation_communities(g)) ###coverage###
nx_com.performance(g, nx_com.label_propagation_communities(g)) ###performance###

from deepsnap.dataset import GraphDataset
from deepsnap.hetero_graph import HeteroGraph
import torch
import warnings
from pylab import *
from copy import deepcopy
from deepsnap.graph import Graph
from deepsnap.batch import Batch

Graph.add_node_attr(g, 'node_feature', 'node_label')

task = 'link_pred'
dg = Graph(g)
# Transform to DeepSNAP dataset
dataset = GraphDataset([dg], task = task, edge_train_mode="all")
print("Original DeepSNAP dataset has {} edges".format(dataset.num_edges[0]))
print("Original DeepSNAP dataset has {} nodes".format(dataset.num_nodes[0]))

# Split the dataset
dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.8, 0.1, 0.1])
num_train_edges = dataset_train[0].edge_label_index.shape[1]
num_val_edges = dataset_val[0].edge_label_index.shape[1]
num_test_edges = dataset_test[0].edge_label_index.shape[1]

edge_color = {}
for i in range(num_train_edges):
  edge = dataset_train[0].edge_label_index[:, i]
  edge = (edge[0].item(), edge[1].item())
  edge_color[edge] = 'darkblue'

for i in range(num_val_edges):
  edge = dataset_val[0].edge_label_index[:, i]
  edge = (edge[0].item(), edge[1].item())
  edge_color[edge] = 'red'

for i in range(num_test_edges):
  edge = dataset_test[0].edge_label_index[:, i]
  edge = (edge[0].item(), edge[1].item())
  edge_color[edge] = 'green'

H = deepcopy(g)

nx.classes.function.set_edge_attributes(H, edge_color, name = 'color')
colors = nx.get_edge_attributes(H, 'color').values()
plt.figure(figsize=(10, 10))
nx.draw(H, pos = pos, cmap = plt.get_cmap('coolwarm'), node_color = 'grey', connectionstyle='arc3, rad = 0.1')
plt.show()

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid, TUDataset
from torch.utils.data import DataLoader
from torch_geometric.nn import SAGEConv

class LinkPredModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinkPredModel, self).__init__()

        self.conv1 = SAGEConv(input_size, hidden_size)
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch):
        x, edge_index, edge_label_index = batch.node_feature, batch.edge_index, batch.edge_label_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)

        nodes_first = torch.index_select(x, 0, edge_label_index[0,:].long())
        nodes_second = torch.index_select(x, 0, edge_label_index[1,:].long())
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        return pred
    
    def loss(self, pred, label):
        return self.loss_fn(pred, label)

def train(model, dataloaders, optimizer, args):
    val_max = 0
    best_model = model

    for epoch in range(1, args["epochs"]):
        for i, batch in enumerate(dataloaders['train']):
            
            batch.to(args["device"])
            model.train()
            optimizer.zero_grad()
            pred = model(batch)
            loss = model.loss(pred, batch.edge_label.type(pred.dtype))

            loss.backward()
            optimizer.step()

            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Loss: {:.5f}'
            score_train = test(model, dataloaders['train'], args)
            score_val = test(model, dataloaders['val'], args)
            score_test = test(model, dataloaders['test'], args)

            print(log.format(epoch, score_train, score_val, score_test, loss.item()))
            if val_max < score_val:
                val_max = score_val
                best_model = copy.deepcopy(model)
    return best_model

def test(model, dataloader, args):
    model.eval()
    score = 0
    num_batches = 0
    for batch in dataloader:
        batch.to(args["device"])
        pred = model(batch)
        pred = torch.sigmoid(pred)
        score += roc_auc_score(batch.edge_label.flatten().cpu().numpy(), pred.flatten().data.cpu().numpy())
        num_batches += 1
    score /= num_batches 
    return score

args = {
    "device" : 'cuda' if torch.cuda.is_available() else 'cpu',
    "hidden_dim" : 128,
    "epochs" : 150,
}

pyg_dataset = Planetoid('./tmp/cora', 'Cora')
graphs = GraphDataset.pyg_to_graphs(pyg_dataset)

dataset = GraphDataset(
        graphs,
        task='link_pred',
        edge_train_mode="all"
    )
datasets = {}
datasets['train'], datasets['val'], datasets['test']= dataset.split(
            transductive=True, split_ratio=[0.85, 0.05, 0.1])
input_dim = datasets['train'].num_node_features
num_classes = datasets['train'].num_edge_labels

model = LinkPredModel(input_dim, args["hidden_dim"]).to(args["device"])

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

dataloaders = {split: DataLoader(
            ds, collate_fn=Batch.collate([]),
            batch_size=1, shuffle=(split=='train'))
            for split, ds in datasets.items()}
best_model = train(model, dataloaders, optimizer, args)
log = "Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
best_train_roc = test(best_model, dataloaders['train'], args)
best_val_roc = test(best_model, dataloaders['val'], args)
best_test_roc = test(best_model, dataloaders['test'], args)
print(log.format(best_train_roc, best_val_roc, best_test_roc))






















