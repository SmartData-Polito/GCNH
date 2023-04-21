import scipy.sparse as sp
import torch
import numpy as np
import pickle as pkl
import sys
import networkx as nx
from dataset import CustomDataset
import argparse
import random
from os import path as path

"""
READ ARGUMENTS 
"""


def parse_boolean(value):
    """Parse boolean values passed as argument"""
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


def parse_args():
    """ Parse arguments """
    parse = argparse.ArgumentParser()

    ## Run details
    parse.add_argument("--model", help="model to train and test", type=str, default="GATH")
    parse.add_argument("-d", "--dataset", help="dataset", type=str, default="cornell")
    parse.add_argument('--model_name', type=str, help='Name of model used', default="Empty")
    parse.add_argument('--verbose', type=parse_boolean, default=False, help='Whether to display training losses')
    parse.add_argument('--hom_syn', type=str, default="h0.00-r1", help='Homophily level for synthetic dataset')
    parse.add_argument('--use_seed', type=parse_boolean, default=True, help='Whether to use seed')
    parse.add_argument('--seed', type=int, default=112, help='Seed')
    parse.add_argument('--splits', type=int, default=0, help='Dataset split') ## Fix this later
    parse.add_argument('--aggfunc', type=str, default="sum", help='Neighbor aggregation function: one of sum, mean or maxpool')

    ## Hyperparameters
    parse.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parse.add_argument('--patience', type=int, default=1000, help='Patience')
    parse.add_argument('--batch_size', type=int, default=100000, help='Batch size')
    parse.add_argument('--nhid', type=int, default=16, help='Hidden size')
    parse.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parse.add_argument('--nlayers', type=int, default=1, help='Number of layers')
    parse.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parse.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay')
    
    args = parse.parse_args()
    return args

"""
GET INFORMATION FOR SELECTED MODEL
"""    
        
def get_nodes_classes(dataset):
    """Get number of nodes and number of classes for the current graph"""

    nodes = {"cornell": 183, "texas":183, "wisconsin":251, "film":7600,
            "chameleon":2277, "squirrel":5201, "cora":2708, "citeseer":3327}

    classes = {"cornell": 5, "texas":5, "wisconsin":5, "film":5,
            "chameleon":5, "squirrel":5, "cora":7, "citeseer":6}

    if dataset not in nodes:
        print("Dataset is not present!")
        return None, None

    return nodes[dataset], classes[dataset]
    

def accuracy(output, labels):
    """Compute accuracy of predictions"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(adj, is_sparse=False):
    """Symmetrically normalize adjacency matrix."""
    if is_sparse:
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()

        adj = sp.coo_matrix((values, (indices[0],indices[1])), shape=adj.shape)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        mx = sparse_mx_to_torch_sparse_tensor(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo())
        assert (mx.coalesce().indices() == indices).all()
        return mx
    else:
        d = adj.sum(dim=1) + 1e-6
        adj = adj / d.view([len(d),1])
        return adj


"""
LOAD GRAPHS FROM FILES
"""

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_idx(split, dataset, labeled):
    """Return fixed splits, doesn't work for syn"""
    idx = np.load("./data/{}/splits/{}_split_0.6_0.2_{}.npz".format(dataset, dataset, split)) 

    if labeled is None:
        idx_train = np.where(idx['train_mask'] == 1)[0]
        idx_test = np.where(idx['test_mask'] == 1)[0]
        idx_val = np.where(idx['val_mask'] == 1)[0]
    else:
        idx_train = np.where(idx['train_mask'][labeled] == 1)[0]
        idx_test = np.where(idx['test_mask'][labeled] == 1)[0]
        idx_val = np.where(idx['val_mask'][labeled] == 1)[0]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
        
    return idx_train, idx_val, idx_test


def load_syn_cora(name):
    
    """Load the dataset in file `syn-cora/<name>.npz`
    `seed` controls the generation of training, validation and test splits"""
    dataset = CustomDataset(root="syn-cora", name=name, setting="gcn", seed=15)

    adj = dataset.adj # Access adjacency matrix
    features = dataset.features # Access node features
    labels = dataset.labels

    idx = np.arange(features.shape[0])
    random.seed(155)
    random.shuffle(idx)
    idx_train = idx[:int(0.5*len(idx))]
    idx_val = idx[int(0.5*len(idx)):int(0.7*len(idx))]
    idx_test = idx[int(0.7*len(idx)):]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)
    features = sp.csr_matrix(features, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    adj = torch.FloatTensor(np.array(adj.todense()))

    return features, labels, adj, idx_train, idx_val, idx_test


def load_data(dataset, split_name=0):
    """Load features, labels and splits for the specified dataset"""
    feature_list = []
    label_list = []
    f = open('./data/{}/out1_node_feature_label.txt'.format(dataset), 'r')
    for line in f.readlines():
        ele = line.strip().split('\t')
        if ele[0] == 'node_id':
            continue
        feature = ele[1]
        label = int(ele[2])
        if dataset == 'film':
            feature_array = np.zeros([931])
            for f in feature.strip().split(','):
                feature_array[int(f)-1] = 1
            feature_list.append(feature_array)
        else:
            feature = feature.strip().split(',')
            feature_list.append(feature)
        label_list.append(label)
    feature = np.array(feature_list, dtype=float)
    idx = np.load("./data/{}/splits/{}_split_0.6_0.2_{}.npz".format(dataset, dataset, split_name)) 
    idx_train = np.where(idx['train_mask'] == 1)[0]
    idx_test = np.where(idx['test_mask'] == 1)[0]
    idx_val = np.where(idx['val_mask'] == 1)[0]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(label_list)
    features = sp.csr_matrix(feature, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    return features, labels, idx_train, idx_val, idx_test

def load_graph(dataset, n_nodes, features=None, undirected=False):
    """Load adjacency matrix for the specified dataset"""
    print('Loading {} dataset...'.format(dataset))

    struct_edges = np.genfromtxt("./data/" + dataset + "/out1_graph_edges.txt", dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(n_nodes, n_nodes),
                         dtype=np.float32)
    if undirected:
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = torch.FloatTensor(np.array(sadj.todense()))

    return nsadj


def load_data_cit(dataset_str, split_name=0, undirected=False):
    """
    Load citation graphs Cora, Citeseer and Pubmed
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if undirected:
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = torch.FloatTensor(np.array(adj.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    lab = torch.zeros([adj.shape[0], ], dtype=torch.long)
    labeled = []
    for i in range(adj.shape[0]):
        if len(np.where(labels[i,:] == 1)[0]) != 0:
            lab[i] = int(np.where(labels[i,:] == 1)[0])
            labeled.append(i)

    adj = torch.reshape(adj[labeled, :][:,labeled], [len(labeled), len(labeled)])
    features = features[labeled, :]
    lab = lab[labeled]

    idx = np.load("./data/{}/splits/{}_split_0.6_0.2_{}.npz".format(dataset_str, dataset_str, split_name))

    idx_train = np.where(idx['train_mask'][labeled] == 1)[0]
    idx_test = np.where(idx['test_mask'][labeled] == 1)[0]
    idx_val = np.where(idx['val_mask'][labeled] == 1)[0]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)
    features = sp.csr_matrix(features, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    
    if dataset_str != "citeseer":
        labeled = None

    return adj, features, lab, idx_train, idx_val, idx_test, labeled

