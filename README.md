# DLMRL:Enhancing Molecular Property Prediction with Dual-Level Representation Learning
Mining molecular graphs at brics level
## Dataset
All datasets we use are from [OGB](https://ogb.stanford.edu/docs/graphprop/).
### Dataset Preprocess
We used the method in [moleood](https://github.com/yangnianzu0515/MoleOOD) to obtain molecular datasets with BRICS substructures.
## Package Dependency
```
torch: 1.9.0
numpy: 1.21.2
ogb: 1.3.4
rdkit: 2021.9.4
scikit-learn: 1.0.2
pyg: 2.0.3
```
## Run the Code
```python main_causal.py```
