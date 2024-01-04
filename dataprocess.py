import os
import pandas
import pickle

import torch
import numpy as np
from functools import reduce
from ogb.utils import smiles2graph
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import BRICS
import re

def motif_dataset(dataset_name):
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader
    
    work_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(work_dir,'dataset')
    dataset = PygGraphPropPredDataset(dataset_name, root = data_dir)
    # 获取数据集文件夹路径
    data_name = dataset_name.replace('-','_')
    dataset_dir = os.path.join(work_dir,'dataset',data_name)
    # 获取smiles
    original_data = pandas.read_csv(
        os.path.join(dataset_dir,'mapping','mol.csv.gz'),
        compression='gzip'
    )
    smiles = original_data.smiles
    
    pre_name = os.path.join(work_dir,'preprocess',data_name)
    pre_file = os.path.join(pre_name,'substructures.pkl')
    
    if not os.path.exists(pre_file):
        raise IOError('Please run preprocess script for this dataset.')
    with open(pre_file,'rb') as Fin:
        substructures = pickle.load(Fin)

    return smiles,substructures,dataset

def graph_from_substructure(subs, return_mask=False, return_type='numpy'):
    sub_struct_list = list(reduce(lambda x, y: x.update(y) or x, subs, set()))
    sub_to_idx = {x: idx for idx, x in enumerate(sub_struct_list)}
    mask = np.zeros([len(subs), len(sub_struct_list)], dtype=bool)
    sub_graph = [smiles2graph(x) for x in sub_struct_list]
    
    # 标识原始子结构包含唯一子结构集合中的哪些元素
    for idx, sub in enumerate(subs):
        mask[idx][list(sub_to_idx[t] for t in sub)] = True

    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    for idx, graph in enumerate(sub_graph):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        # 'node_feat': np.concatenate(node_feats, axis=0),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }

    assert return_type in ['numpy', 'torch', 'pyg'], 'Invaild return type'
    if return_type in ['torch', 'pyg']:
        for k, v in result.items():
            result[k] = torch.from_numpy(v)

    result['num_nodes'] = lstnode

    if return_type == 'pyg':
        result = Data(**result)

    if return_mask:
        return result, mask
    else:
        return result

def draw_explain_graph(smiles, subs, masks, sub_masks, threshold, global_id):
    if not os.path.exists('graphs/mols'):
        os.makedirs('graphs/mols')
    if not os.path.exists('graphs/subs'):
        os.makedirs('graphs/subs')

    for idx in range(len(smiles)):
        cur_smiles = smiles[idx]
        brics_fragments = subs[idx]
        mol = Chem.MolFromSmiles(cur_smiles)

        # 绘制分子子结构图
        # frags = BRICS.BreakBRICSBonds(mol)
        # pieces=Chem.GetMolFrags(frags,asMols=True)
        # subs_img = Draw.MolsToGridImage(pieces)
        # subs_img_path = f'graphs/subs/{global_id + idx}.png'
        # subs_img.save(subs_img_path)

        # 获取当前分子图子结构mask
        sub_indices = torch.from_numpy(masks[idx])
        sub_indices_idx = torch.where(sub_indices)[0]
        cur_subgraph_mask = sub_masks[sub_indices_idx].cpu()

        # 根据子结构重要程度设置节点颜色
        highlight_atoms = []
        highlight_bonds = []

        is_painted = False

        for frag, importance in zip(brics_fragments, cur_subgraph_mask):
            if importance > threshold:
                is_painted = True
                frag_no_tags = remove_brics_tags(frag)
                if frag_no_tags:
                    submol = Chem.MolFromSmarts(frag_no_tags)
                    atoms = mol.GetSubstructMatches(submol)

                    for atom_group in atoms:
                        highlight_atoms.extend(atom_group)
                        for i in range(len(atom_group)-1):
                            bond = mol.GetBondBetweenAtoms(atom_group[i], atom_group[i+1])
                            if bond:
                                highlight_bonds.append(bond.GetIdx())
                else:
                    print(f"Invalid SMARTS after removing BRICS tags: {frag}")
            
        if is_painted:
            # 绘制分子子结构图
            frags = BRICS.BreakBRICSBonds(mol)
            pieces=Chem.GetMolFrags(frags,asMols=True)
            subs_img = Draw.MolsToGridImage(pieces)
            subs_img_path = f'graphs/subs/{global_id + idx}.png'
            subs_img.save(subs_img_path)
            img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds, size=(300, 300))
            img_path = f'graphs/mols/{global_id + idx}.png'
            img.save(img_path)

def remove_brics_tags(smiles):
    # 先移除BRICS标记
    modified_smiles = re.sub(r'\[\d+\*\]', '', smiles)
    # 再移除空括号
    modified_smiles = re.sub(r'\(\)', '', modified_smiles)
    # 确保处理后的字符串不是空的
    return modified_smiles if modified_smiles.strip() else None

def draw_brics_substructures(smiles):
    mol = Chem.MolFromSmiles(smiles)
    brics_fragments = BRICS.BreakBRICSBonds(mol)

    sub_mols = [Chem.MolFromSmiles(frag) for frag in brics_fragments]

    img = MolsToGridImage(sub_mols, molsPerRow = 3, subImgSize=(200, 200), legends=[Chem.MolToSmiles(mol) for mol in sub_mols])
    img_path = f'graphs/subs/{global_id + idx}.png'
    img.save(img_path)