import torch
from tqdm import tqdm

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train_one_epoch(model, device, loader, optimizer, task_type, alpha):
    model.train()

    for step, (smiles,subs,graphs) in enumerate(tqdm(loader, desc="Iteration")):
        subs = [eval(x) for x in subs]
        graphs = graphs.to(device)
        if graphs.x.shape[0] == 1 or graphs.batch[-1] == 0:
            pass
        else:
            pred, cosine_loss, *_ = model(smiles,graphs,subs)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = graphs.y == graphs.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], graphs.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], graphs.y.to(torch.float32)[is_labeled])
            total_loss = (1 - alpha) * loss + alpha * cosine_loss
            total_loss.backward()
            optimizer.step()

def eval_one_epoch(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, (smiles,subs,graphs) in enumerate(tqdm(loader, desc="Iteration")):
        subs = [eval(x) for x in subs]
        graphs = graphs.to(device)

        if graphs.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, cosine_loss, *_ = model(smiles,graphs,subs)

            y_true.append(graphs.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)