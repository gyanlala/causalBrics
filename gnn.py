import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode

from dataprocess import graph_from_substructure

class DualGNN(torch.nn.Module):
    def __init__(self, num_tasks, num_layer = 4, emb_dim = 256, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.1, JK = "last", graph_pooling = "mean"):
        
        super(DualGNN,self).__init__()
        # 全图GNN
        self.gnn = BaseGNN(num_tasks,num_layer,emb_dim,gnn_type,virtual_node,residual,drop_ratio,JK,graph_pooling)
        # 子结构GNN
        self.sub_gnn = BaseGNN(num_tasks,5,emb_dim,gnn_type,virtual_node,residual,0.3,JK,graph_pooling)
        
        self.combined_linear = torch.nn.Linear(2*emb_dim,num_tasks)
        # self.combined_linear = torch.nn.Linear(emb_dim,num_tasks)
    
    def feature_from_subs(self,subs,device,return_mask=False):
        substructure_graph,mask = graph_from_substructure(subs,return_mask,'pyg')
        substructure_graph = substructure_graph.to(device)
        h_sub = self.sub_gnn(substructure_graph)
        return h_sub,mask
        
        
    def forward(self,graphs,subs,aggr="mean"):
        h_graph = self.gnn(graphs)
        h_sub, mask = self.feature_from_subs(subs=subs,device=graphs.x.device,return_mask=True)

        h_sub_aligned = torch.zeros_like(h_graph)

        for idx, sub_indices in enumerate(mask):
            if aggr == "sum":
                h_sub_aligned[idx] += h_sub[sub_indices].sum(dim=0)
            elif aggr == "mean":
                h_sub_aligned[idx] += h_sub[sub_indices].mean(dim=0)

        h_combined = torch.cat([h_graph,h_sub_aligned],dim=1)

        return self.combined_linear(h_combined)
        # return self.combined_linear(h_sub_aligned)

class AttentionAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim):
        super(AttentionAgger, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Kdim, Mdim)

    def forward(self, Q, K, V, mask=None):
        Q, K = self.WQ(Q), self.WK(K)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))
        Attn = torch.softmax(Attn, dim=-1)
        return torch.matmul(Attn, V)
        
class DualGNN1(torch.nn.Module):
    def __init__(self, num_tasks, num_layer = 4, emb_dim = 256, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.1, JK = "last", graph_pooling = "mean"):
        
        super(DualGNN1,self).__init__()
        # 全图GNN
        self.gnn = BaseGNN(num_tasks,num_layer,emb_dim,gnn_type,virtual_node,residual,drop_ratio,JK,graph_pooling)
        # 子结构GNN
        self.sub_gnn = BaseGNN(num_tasks,num_layer,emb_dim,gnn_type,virtual_node,residual,drop_ratio,JK,graph_pooling)
        # 全图与子结构的特征维度待确认
        self.attenaggr = AttentionAgger(emb_dim,emb_dim,emb_dim)
        
        # self.predictor = torch.nn.Linear(emb_dim,num_tasks)
        self.predictor = torch.nn.Linear(2*emb_dim,num_tasks)
    
    def feature_from_subs(self,subs,device,return_mask=False):
        substructure_graph,mask = graph_from_substructure(subs,return_mask,'pyg')
        substructure_graph = substructure_graph.to(device)
        h_sub = self.sub_gnn(substructure_graph)
        return h_sub,mask
        
        
    def forward(self,graphs,subs):
        h_graph = self.gnn(graphs)
        h_sub, mask = self.feature_from_subs(subs=subs,device=graphs.x.device,return_mask=True)

        mask = torch.from_numpy(mask).to(graphs.x.device)
        Attn_mask = torch.logical_not(mask)
        
        # 使用注意力机制合并全图和子结构特征
        molecule_feat = self.attenaggr(Q=h_graph,K=h_sub,V=h_sub,mask=Attn_mask)
        h_combined = torch.cat([h_graph,molecule_feat],dim=1)

        # return self.predictor(molecule_feat)  
        return self.predictor(h_combined)        
    
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