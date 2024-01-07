import torch
import torch.nn as nn

import math
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode

from dataprocess import graph_from_substructure, draw_explain_graph

class SubMaskGenerator(torch.nn.Module):
    def __init__(self,emb_dim):
        super(SubMaskGenerator,self).__init__()
        self.mask_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_dim,2*emb_dim),
            # torch.nn.Linear(emb_dim,4*emb_dim),
            # torch.nn.BatchNorm1d(4*emb_dim),
            # torch.nn.LeakyReLU(),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            # torch.nn.Linear(4*emb_dim,2*emb_dim),
            # torch.nn.LeakyReLU(),
            torch.nn.Linear(2*emb_dim,1)
        )
    
    def forward(self,subgraph_features):
        mask = self.mask_nn(subgraph_features).squeeze()
        return torch.sigmoid(mask)
        
class CausalGNN(torch.nn.Module):
    def __init__(self, num_tasks, num_layer = 4, sub_num_layer = 3, emb_dim = 256, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.3, sub_drop_ratio = 0.1, JK = "last", graph_pooling = "mean", threshold = 0.4):
        
        super(CausalGNN,self).__init__()
        # 全图GNN
        self.gnn = BaseGNN(num_tasks,num_layer,emb_dim,gnn_type,virtual_node,residual,drop_ratio,JK,"sum")
        # 子结构GNN
        self.sub_gnn = BaseGNN(num_tasks,sub_num_layer,emb_dim,gnn_type,virtual_node,residual,sub_drop_ratio,JK,"sum")
        # 子结构mask
        self.sub_mask_generator = SubMaskGenerator(emb_dim)
        # BRICS过滤值
        self.threshold = threshold
        self.combined_linear = torch.nn.Linear(2*emb_dim,num_tasks)
    
    def feature_from_subs(self,subs,device,return_mask=False):
        substructure_graph, mask = graph_from_substructure(subs,return_mask,'pyg')
        substructure_graph = substructure_graph.to(device)
        h_sub = self.sub_gnn(substructure_graph)
        return h_sub,mask
        
        
    def forward(self,smiles,graphs,subs,aggr="mean"):
        h_graph = self.gnn(graphs)
        h_sub, mask = self.feature_from_subs(subs=subs,device=graphs.x.device,return_mask=True)
        
        # 生成子图掩码
        subgraph_mask = self.sub_mask_generator(h_sub)
        h_sub_aligned = torch.zeros_like(h_graph)
        
        for idx, sub_indices in enumerate(mask):
            if isinstance(sub_indices,np.ndarray):
                sub_indices = torch.from_numpy(sub_indices).to(graphs.x.device)
            # 计算每个分子图的有效子结构索引
            sub_indices_idx = torch.where(sub_indices)[0]
            cur_subgraph_mask = subgraph_mask[sub_indices_idx].cpu()

            valid_sub_indices = sub_indices_idx[cur_subgraph_mask > self.threshold]

            if len(valid_sub_indices) > 0:
                if aggr == "sum":
                    h_sub_aligned[idx] += h_sub[valid_sub_indices].sum(dim=0)
                elif aggr == "mean":
                    h_sub_aligned[idx] += h_sub[valid_sub_indices].mean(dim=0)
            else:
                h_sub_aligned[idx] += torch.zeros_like(h_sub[0])
                # h_sub_aligned[idx] += h_graph[idx]

        # 计算 h_graph 与 h_sub_aligned 的余弦相似度，进行特征空间的对齐
        # 规范化 h_sub_aligned 和 h_graph
        h_sub_aligned_norm = F.normalize(h_sub_aligned, p=2, dim=1)
        h_graph_norm = F.normalize(h_graph, p=2, dim=1)

        cosine_sim = F.cosine_similarity(h_sub_aligned_norm,h_graph_norm)
        cosine_loss = 1 - cosine_sim.mean()

        h_combined = torch.cat([h_graph,h_sub_aligned],dim=1)

        return self.combined_linear(h_combined), cosine_loss, mask, subgraph_mask
    
class BaseGNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 4, emb_dim = 256, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.1, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(BaseGNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        # return self.graph_pred_linear(h_graph)
        return h_graph

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 4, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)

if __name__ == '__main__':
    GNN(num_tasks = 10)