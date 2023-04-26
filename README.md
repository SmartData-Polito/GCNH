# GCNH: A Simple Method For Representation Learning On Heterophilous Graphs

This repository contains the implementation of Graph Convolutional Network for Heterophily, the model described in our work *GCNH: A Simple Method For Representation Learning On Heterophilous Graphs*, accepted at IJCNN 2023 ([preprint](https://arxiv.org/abs/2304.10896)). 

## Description

GCNH extends Graph Neural networks' representation capabilities on by learning two different functions to encode center node and neighborhood messages. These two encoding are merged into the final node embedding using a convex combination with a learned coefficient β. The layer propagation rule in GCNH works as follows: 

$$h^l_u = (1-\beta)\bigoplus_{v \in \mathcal{N_u}}\left[ \sigma(h^{\ell-1}_v W_2) \right] + \beta\sigma(h^{\ell-1}_u W_1).$$

For a center node, GCNH flexibly assigns more or less importance to the neighborhood, depending on how informative neighbors are - this design greatly improves learning on **heterophilous** graphs. Experiments show that GCNH achieves state-of-the-art performance on 4 out of the 8 graph datasets used.

![GCNH_layer](./figures/gcnh_layer_background.svg)  

## Usage

### Requirements
 * Python=3.9
 * requirements.txt (`pip install -r requirements.txt`)
 * [torch-scatter](https://pypi.org/project/torch-scatter/) (e.g., `pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.0+{device}.html`)

The folder [GCNH/experiments](https://github.com/SmartData-Polito/GCNH/tree/main/experiments) contains the commands to reproduce the results of the main table of the paper and to test GCNH on the synthetic dataset used in the paper.

## Contributors

- [Andrea Cavallo](https://github.com/andrea-cavallo-98)
- [Claas Grohnfeldt](https://github.com/claas-grohnfeldt)
- [Michele Russo](https://github.com/mik1904)

## Citation

If you find this code useful, please cite

```
@INPROCEEDINGS{Cava2306:GCNH,
AUTHOR="Andrea Cavallo and Luca Vassio and Claas Grohnfeldt and Michele Russo and Giulio Lovisotto",
TITLE="{GCNH:} A Simple Method for Representation Learning on Heterophilous Graphs",
BOOKTITLE="2023 International Joint Conference on Neural Networks (IJCNN) (IJCNN 2023)",
ADDRESS="Queensland, Australia",
PAGES=8,
DAYS="17",
MONTH=jun,
YEAR=2023,
}
```
