import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from causalgnn import CausalGNN

from tqdm import tqdm
import argparse
import time
import numpy as np

import os
import json
import datetime

### importing OGB
from dataprocess import motif_dataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train_one_epoch(model, device, loader, optimizer, task_type, alpha):
    model.train()

    for step, (subs,graphs) in enumerate(tqdm(loader, desc="Iteration")):
        subs = [eval(x) for x in subs]
        graphs = graphs.to(device)
        if graphs.x.shape[0] == 1 or graphs.batch[-1] == 0:
            pass
        else:
            pred, cosine_loss = model(graphs,subs)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = graphs.y == graphs.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], graphs.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], graphs.y.to(torch.float32)[is_labeled])
            total_loss = loss + alpha * cosine_loss
            total_loss.backward()
            optimizer.step()

def eval_one_epoch(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (subs,graphs) in enumerate(tqdm(loader, desc="Iteration")):
        subs = [eval(x) for x in subs]
        graphs = graphs.to(device)

        if graphs.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, cosine_loss = model(graphs,subs)

            y_true.append(graphs.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.1,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--alpha', type=int, default=0.5,
                        help='weight for cosine similarity loss')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()
    
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    auto_filename = f'logs/feature_aligned/results_{current_time}.txt'
    
    if not args.filename:
        args.filename = auto_filename

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    # dataset = PygGraphPropPredDataset(name = args.dataset)
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
    
    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]
    train_subs = [str(total_subs[x.item()]) for x in train_idx]
    valid_subs = [str(total_subs[x.item()]) for x in valid_idx]
    test_subs = [str(total_subs[x.item()]) for x in test_idx]

    train_loader = DataLoader(list(zip(train_subs,train_dataset)), batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(list(zip(valid_subs,valid_dataset)), batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(list(zip(test_subs,test_dataset)), batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    
    if args.gnn == 'gin':
        model = CausalGNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = CausalGNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = CausalGNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = CausalGNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

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

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

        
    if args.filename:
        save_dict = {
            'Val': valid_curve[best_val_epoch],
            'Test': test_curve[best_val_epoch],
            'Train': train_curve[best_val_epoch],
            'BestTrain': best_train,
            'Config': {
                'Device': args.device,
                'GNN': args.gnn,
                'Drop ratio': args.drop_ratio,
                'Number of layers': args.num_layer,
                'Embedding dimension': args.emb_dim,
                'Batch size': args.batch_size,
                'Epochs': args.epochs,
                'Number of workers': args.num_workers,
                'Dataset': args.dataset,
                'Feature': args.feature
            }
        }
        with open(args.filename, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)

    print(f"Results and configurations have been saved to {args.filename}")


if __name__ == "__main__":
    main()