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
        mask = self.mask_nn(subgraph_features).squeeze(-1)
        return torch.sigmoid(mask)
        
class CausalGNN(torch.nn.Module):
    def __init__(self, num_tasks, num_layer = 4, sub_num_layer = 3, emb_dim = 256, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.3, sub_drop_ratio = 0.1, JK = "last", graph_pooling = "mean", threshold = 0.4, margin = 1.0):
        
        super(CausalGNN,self).__init__()
        # 全图GNN
        self.gnn = BaseGNN(num_tasks,num_layer,emb_dim,gnn_type,virtual_node,residual,drop_ratio,JK,"mean")
        # 子结构GNN
        self.sub_gnn = BaseGNN(num_tasks,sub_num_layer,emb_dim,gnn_type,virtual_node,residual,sub_drop_ratio,JK,"mean")
        # 子结构mask
        self.sub_mask_generator = SubMaskGenerator(emb_dim)
        # 正负样本间隔
        self.margin = margin
        # BRICS过滤值
        self.threshold = threshold
        # self.combined_linear = torch.nn.Linear(emb_dim,num_tasks)
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
        h_sub_env = torch.zeros_like(h_graph)

        if self.threshold == 1 :
            h_combined = torch.cat([h_graph,h_sub_aligned],dim=1)
            return self.combined_linear(h_combined), 0, mask, subgraph_mask
            # return self.combined_linear(h_graph), 0, mask, subgraph_mask
        
        for idx, sub_indices in enumerate(mask):
            if isinstance(sub_indices,np.ndarray):
                sub_indices = torch.from_numpy(sub_indices).to(graphs.x.device)
            # 计算每个分子图的有效子结构索引
            sub_indices_idx = torch.where(sub_indices)[0]

            cur_subgraph_mask = subgraph_mask[sub_indices_idx].cpu()

            valid_sub_indices = sub_indices_idx[cur_subgraph_mask > self.threshold]
            invalid_sub_indices = sub_indices_idx[cur_subgraph_mask <= (self.threshold - 0.1)]

            if len(valid_sub_indices) > 0:
                if aggr == "sum":
                    h_sub_aligned[idx] += h_sub[valid_sub_indices].sum(dim=0)
                elif aggr == "mean":
                    h_sub_aligned[idx] += h_sub[valid_sub_indices].mean(dim=0)
            else:
                h_sub_aligned[idx] += torch.zeros_like(h_sub[0])
 
            if len(invalid_sub_indices) > 0:
                if aggr == "sum":
                    h_sub_env[idx] += h_sub[invalid_sub_indices].sum(dim=0)
                elif aggr == "mean":
                    h_sub_env[idx] += h_sub[invalid_sub_indices].mean(dim=0)
            else:
                h_sub_env[idx] += torch.zeros_like(h_sub[0])

        # 对比学习损失
        # 有效子结构表征的相似度
        batch_size = graphs.batch.max().item() + 1
        contrastive_loss = 0

        valid_sim_matrix = 1 - F.cosine_similarity(h_sub_aligned.unsqueeze(1), h_sub_aligned.unsqueeze(0), dim=2)
        non_zero_pos_indices = torch.any(h_sub_aligned != 0, dim=1)
        pos_num = non_zero_pos_indices.sum() - 1

        # print(h_sub_aligned)

        # 过滤负样本
        non_zero_neg_indices = torch.any(h_sub_env != 0, dim=1)
        filtered_h_sub_env = h_sub_env[non_zero_neg_indices]
        # print(filtered_h_sub_env)
        # print(filtered_h_sub_env.size(0))

        for i in range(batch_size):
            cur_sub_repr = h_sub_aligned[i].unsqueeze(0)
            if torch.all(cur_sub_repr == 0):
                continue

            # 计算平均负样本距离
            if filtered_h_sub_env.size(0) == 0:
                continue

            neg_distances = 1 - F.cosine_similarity(cur_sub_repr, filtered_h_sub_env, dim=1)
            negative_sample = neg_distances.sum() / filtered_h_sub_env.size(0)
                
            # 计算平均正样本距离
            pos_distances = valid_sim_matrix[i]
            positive_sample = pos_distances.sum() / pos_num

            # print("positive sample:{},negative_sample:{}".format(positive_sample , negative_sample))

            contrastive_loss += F.relu(positive_sample - negative_sample + self.margin)

        contrastive_loss /= batch_size

        h_combined = torch.cat([h_graph,h_sub_aligned],dim=1)

        return self.combined_linear(h_combined), contrastive_loss, mask, subgraph_mask

    # INFONCE Loss
    # def forward(self, smiles, graphs, subs, aggr="mean"):
    #     h_graph = self.gnn(graphs)
    #     h_sub, mask = self.feature_from_subs(subs=subs,device=graphs.x.device,return_mask=True)
        
    #     # 生成子图掩码
    #     subgraph_mask = self.sub_mask_generator(h_sub)
    #     h_sub_aligned = torch.zeros_like(h_graph)
    #     h_sub_env = torch.zeros_like(h_graph)

    #     if self.threshold == 1 :
    #         h_combined = torch.cat([h_graph,h_sub_aligned],dim=1)
    #         return self.combined_linear(h_combined), 0, mask, subgraph_mask
    #         # return self.combined_linear(h_graph), 0, mask, subgraph_mask
        
    #     for idx, sub_indices in enumerate(mask):
    #         if isinstance(sub_indices,np.ndarray):
    #             sub_indices = torch.from_numpy(sub_indices).to(graphs.x.device)
    #         # 计算每个分子图的有效子结构索引
    #         sub_indices_idx = torch.where(sub_indices)[0]
    #         cur_subgraph_mask = subgraph_mask[sub_indices_idx].cpu()

    #         valid_sub_indices = sub_indices_idx[cur_subgraph_mask > self.threshold]
    #         invalid_sub_indices = sub_indices_idx[cur_subgraph_mask <= (self.threshold - 0.1)]

    #         if len(valid_sub_indices) > 0:
    #             if aggr == "sum":
    #                 h_sub_aligned[idx] += h_sub[valid_sub_indices].sum(dim=0)
    #             elif aggr == "mean":
    #                 h_sub_aligned[idx] += h_sub[valid_sub_indices].mean(dim=0)
    #         else:
    #             h_sub_aligned[idx] += torch.zeros_like(h_sub[0])
 
    #         if len(invalid_sub_indices) > 0:
    #             if aggr == "sum":
    #                 h_sub_env[idx] += h_sub[invalid_sub_indices].sum(dim=0)
    #             elif aggr == "mean":
    #                 h_sub_env[idx] += h_sub[invalid_sub_indices].mean(dim=0)
    #         else:
    #             h_sub_env[idx] += torch.zeros_like(h_sub[0])

    #     batch_size = graphs.batch.max().item() + 1
    #     contrastive_loss = 0

    #     # 过滤负样本
    #     non_zero_neg_indices = torch.any(h_sub_env != 0, dim=1)
    #     filtered_h_sub_env = h_sub_env[non_zero_neg_indices]

    #     temperature = 0.4  # 温度参数

    #     for i in range(batch_size):
    #         cur_sub_repr = h_sub_aligned[i].unsqueeze(0)
    #         if torch.all(cur_sub_repr == 0):
    #             continue

    #         # 初始化neg_similarities为一个空张量
    #         neg_similarities = torch.tensor([], device=cur_sub_repr.device)

    #         # 计算与正样本的相似度
    #         pos_similarities = F.cosine_similarity(cur_sub_repr, h_sub_aligned, dim=1) / temperature
            
    #         # 如果有负样本，则计算与负样本的相似度
    #         if filtered_h_sub_env.size(0) > 0:
    #             neg_similarities = F.cosine_similarity(cur_sub_repr, filtered_h_sub_env, dim=1) / temperature

    #         # 二元交叉熵需要目标为0或1
    #         pos_targets = torch.ones_like(pos_similarities)  # 正样本目标设置为1
    #         neg_targets = torch.zeros_like(neg_similarities) # 负样本目标设置为0
            
    #         # 将正负样本相似度和目标合并
    #         all_similarities = torch.cat((pos_similarities, neg_similarities), dim=0) if neg_similarities.nelement() != 0 else pos_similarities
    #         all_targets = torch.cat((pos_targets, neg_targets), dim=0) if neg_similarities.nelement() != 0 else pos_targets

    #         # 计算二元交叉熵损失
    #         bce_loss = F.binary_cross_entropy_with_logits(all_similarities, all_targets)
    #         contrastive_loss += bce_loss

    #     contrastive_loss /= batch_size

    #     h_combined = torch.cat([h_graph, h_sub_aligned], dim=1)

    #     return self.combined_linear(h_combined), contrastive_loss, mask, subgraph_mask
    
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