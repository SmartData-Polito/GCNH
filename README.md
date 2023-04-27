# GCNH: A Simple Method For Representation Learning On Heterophilous Graphs

This repository contains the implementation of GCNH, the model described in our work *GCNH: A Simple Method For Representation Learning On Heterophilous Graphs*, accepted at IJCNN 2023 ([preprint](https://arxiv.org/abs/2304.10896)). 

## Description

GCNH extends GNNs' representation capabilities on heterophilous graphs by learning two different functions to encode center node and neighborhood messages. These two encoding are merged into the final node embedding using a balancing coefficient β. In this way, the model can flexibly choose to assign more or less importance to the neighborhood, depending on how informative it is. Our experiments show that GCNH achieves state-of-the-art performance on 4 out of the 8 graph datasets used.

![GCNH_layer](./figures/gcnh_layer_background.svg)  

## Usage

The folder `experiments` contains the commands to reproduce the results of the main table of the paper and to test GCNH on the synthetic dataset used in the paper.

## Contributors

- [Andrea Cavallo](https://github.com/andrea-cavallo-98)
- [Claas Grohnfeldt](https://github.com/claas-grohnfeldt)
- [Michele Russo](https://github.com/mik1904)
- [Giulio Lovisotto](https://giuliolovisotto.github.io)

## Citation

If you find this code useful, please cite

```
@inproceedings{cavallo2023gcnh,
  title={{GCNH:} A Simple Method for Representation Learning on Heterophilous Graphs},
  author={Cavallo, Andrea and Grohnfeldt, Claas and Russo, Michele and Lovisotto, Giulio and Vassio, Luca},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)},
  year={2022},
  organization={IEEE}
}
```
