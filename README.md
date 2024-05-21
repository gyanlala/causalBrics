# DLMRL:Enhancing Molecular Property Prediction with Dual-Level Representation Learning
## Overview
This repository contains the core implementation of DLMRL, which enhances molecular property prediction by mining molecular graphs at the BRICS level.
## Datasets
All datasets used in our work are from [OGB](https://ogb.stanford.edu/docs/graphprop/).
### Dataset Preprocess
We used the method from [moleood](https://github.com/yangnianzu0515/MoleOOD) to obtain molecular datasets with BRICS substructures.
## Package Dependency
To run the code, you will need the following packages:
```
torch: 1.9.1
numpy: 1.21.2
ogb: 1.3.6
rdkit: 2023.3.3
scikit-learn: 1.0.2
pyg: 2.0.3
```
## Run the Code
To execute the main script with the default configuration, run the following command:
```python main_causal.py```
