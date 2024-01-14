import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from causalgnn import CausalGNN

from tqdm import tqdm
import argparse
import time
import random
import numpy as np

import os
import json
import datetime

### importing OGB
from train import train_one_epoch, eval_one_epoch
from dataprocess import motif_dataset, draw_explain_graph
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# settings
parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--gnn', type=str, default='gcn',
                    help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
parser.add_argument('--drop_ratio', type=float, default=0.4,
                    help='dropout ratio (default: 0.3)')
parser.add_argument('--sub_drop_ratio', type=float, default=0.1,
                    help='sub dropout ratio (default: 0.1)')
parser.add_argument('--num_layer', type=int, default=5,
                    help='number of GNN message passing layers (default: 5)')
parser.add_argument('--sub_num_layer', type=int, default=3,
                    help='number of subGNN message passing layers (default: 3)')
parser.add_argument('--emb_dim', type=int, default=256,
                    help='dimensionality of hidden units in GNNs (default: 256)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='threshold of substructure mask')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='weight for cosine similarity loss')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate of optimizer')
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (default: ogbg-molhiv)')
parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')
parser.add_argument('--filename', type=str, default="",
                    help='filename to output result (default: )')

def load_model(model_path, model_class, device, **model_kwargs):
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def main(args, device):
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    auto_filename = f'logs/ablation/results_{current_time}.txt'
    
    if not args.filename:
        args.filename = auto_filename

    ### automatic dataloading and splitting
    total_smiles,total_subs,dataset = motif_dataset(args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]


    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    total_smiles_list = total_smiles.tolist()
    train_smiles = [total_smiles_list[x.item()] for x in train_idx]
    valid_smiles = [total_smiles_list[x.item()] for x in valid_idx]
    test_smiles = [total_smiles_list[x.item()] for x in test_idx]
    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]
    train_subs = [str(total_subs[x.item()]) for x in train_idx]
    valid_subs = [str(total_subs[x.item()]) for x in valid_idx]
    test_subs = [str(total_subs[x.item()]) for x in test_idx]

    train_loader = DataLoader(list(zip(train_smiles,train_subs,train_dataset)), batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(list(zip(valid_smiles,valid_subs,valid_dataset)), batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(list(zip(test_smiles,test_subs,test_dataset)), batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    
    if args.gnn == 'gin':
        model = CausalGNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, sub_num_layer = args.sub_num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, sub_drop_ratio = args.sub_drop_ratio, virtual_node = False, threshold = args.threshold).to(device)
    elif args.gnn == 'gin-virtual':
        model = CausalGNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, sub_num_layer = args.sub_num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, sub_drop_ratio = args.sub_drop_ratio, virtual_node = True, threshold = args.threshold).to(device)
    elif args.gnn == 'gcn':
        model = CausalGNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, sub_num_layer = args.sub_num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, sub_drop_ratio = args.sub_drop_ratio, virtual_node = False, threshold = args.threshold).to(device)
    elif args.gnn == 'gcn-virtual':
        model = CausalGNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, sub_num_layer = args.sub_num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, sub_drop_ratio = args.sub_drop_ratio, virtual_node = True, threshold = args.threshold).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    valid_curve = []
    test_curve = []
    train_curve = []

    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_val_score = float('inf') if 'regression' in dataset.task_type else 0
    best_test_score = float('inf') if 'regression' in dataset.task_type else 0
    best_model_path = os.path.join(save_dir, f"{args.gnn}_{current_time}.pth")
    best_epoch = 0

    patience = 18
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_one_epoch(model, device, train_loader, optimizer, dataset.task_type, args.alpha)

        print('Evaluating...')
        train_perf = eval_one_epoch(model, device, train_loader, evaluator)
        valid_perf = eval_one_epoch(model, device, valid_loader, evaluator)
        test_perf = eval_one_epoch(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        # 保存最佳模型
        cur_val_score = valid_perf[dataset.eval_metric]
        cur_test_score = test_perf[dataset.eval_metric]
        is_best_model = False

        if 'classification' in  dataset.task_type:
            if cur_val_score > best_val_score and cur_test_score > best_test_score:
                is_best_model = True
                patience_counter = 0
            else:
                patience_counter += 1

        else: #regression
            if cur_val_score < best_val_score and cur_test_score < best_test_score:
                is_best_model = True
                patience_counter = 0
            else:
                patience_counter += 1

        if is_best_model:
            best_val_score = cur_val_score
            best_test_score = cur_test_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path}")

        # 检查是否达到容忍度限制
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}!")
            break

    print('Finished training!')
    print(f"Best validation score: {best_val_score}")
    print(f"Test score: {best_test_score}")
    print(f"Best model saved at {best_model_path}")

        
    if args.filename:
        save_dict = {
            'Val': best_val_score,
            'Test': best_test_score,
            'Train': train_curve[best_epoch - 1],
            'BestEpoch': best_epoch,
            'Config': {
                'Device': args.device,
                'Feature': args.feature,
                'Drop ratio': args.drop_ratio,
                'Sub drop ratio': args.sub_drop_ratio,
                'Number of layers': args.num_layer,
                'Number of sub_layers': args.sub_num_layer,
                'Embedding dimension': args.emb_dim,
                'Batch size': args.batch_size,
                'Threshold': args.threshold,
                'alpha': args.alpha,
                'lr': args.lr,
                'Epochs': args.epochs,
                'Number of workers': args.num_workers,
                'Dataset': args.dataset,
                'GNN': args.gnn
            }
        }
        with open(args.filename, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=1)

    print(f"Results and configurations have been saved to {args.filename}")
    return best_model_path

def explain(args, device, best_model_path):
    ### automatic dataloading and splitting
    total_smiles,total_subs,dataset = motif_dataset(args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    total_smiles_list = total_smiles.tolist()
    total_subs_list = [str(sub) for sub in total_subs]
    total_loader = DataLoader(list(zip(total_smiles_list, total_subs_list, dataset)), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    if args.gnn == 'gin':
        model = CausalGNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, sub_num_layer = args.sub_num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, sub_drop_ratio = args.sub_drop_ratio, virtual_node = False, threshold = args.threshold).to(device)
    elif args.gnn == 'gin-virtual':
        model = CausalGNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, sub_num_layer = args.sub_num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, sub_drop_ratio = args.sub_drop_ratio, virtual_node = True, threshold = args.threshold).to(device)
    elif args.gnn == 'gcn':
        model = CausalGNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, sub_num_layer = args.sub_num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, sub_drop_ratio = args.sub_drop_ratio, virtual_node = False, threshold = args.threshold).to(device)
    elif args.gnn == 'gcn-virtual':
        model = CausalGNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, sub_num_layer = args.sub_num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, sub_drop_ratio = args.sub_drop_ratio, virtual_node = True, threshold = args.threshold).to(device)
    else:
        raise ValueError('Invalid GNN type')

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    for batch_idx, (smiles,subs,graphs) in  enumerate(tqdm(total_loader, desc="Iteration")):
        subs = [eval(x) for x in subs]
        graphs = graphs.to(device)

        if graphs.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                global_id = batch_idx * args.batch_size
                pred, cosine_loss, mask, subgraph_mask = model(smiles,graphs,subs)
                draw_explain_graph(smiles, subs, mask, subgraph_mask, 0.6, global_id)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    best_model_path = main(args,device)
    # explain(args, device, best_model_path)