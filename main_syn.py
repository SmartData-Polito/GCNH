"""
Perform training and testing of GCNH on the synthetic dataset
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
import os
from tqdm import tqdm
from copy import deepcopy
from models import GCNH
from scipy.sparse import coo_matrix


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args = parse_args()

    dataset = "syn"
    n_classes = 5

    cuda = torch.cuda.is_available()

    if args.use_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    for hom_syn in ["h0.00-r", "h0.10-r", "h0.20-r", "h0.30-r", "h0.40-r", "h0.50-r", "h0.60-r", "h0.70-r", "h0.80-r", "h0.90-r", "h1.00-r"]:
        final_acc = []
        b_list = []
        for r in range(1,4): # There are 3 datasets for each homophily level
            print("Loading graph ", hom_syn + str(r))

            features, labels, adj, idx_train, idx_val, idx_test = load_syn_cora(hom_syn + str(r))

            if args.aggfunc == "mean":
                adj = normalize(adj)

            if args.aggfunc == "maxpool":
                # Precomputing this allows for a fast execution of maxpooling aggregation
                coo_m =  coo_matrix(adj.numpy())
                row, col = torch.tensor(coo_m.row).long(), torch.tensor(coo_m.col).long()
            else:
                row, col = None, None

            model = GCNH(nfeat=features.shape[1],
                        nhid=args.nhid,
                        nclass=n_classes,
                        dropout=args.dropout,
                        nlayers=args.nlayers,
                        maxpool=args.aggfunc == "maxpool")
            if cuda:
                model.cuda()
                features = features.cuda()
                adj = adj.cuda()
                labels = labels.cuda()
                idx_train = idx_train.cuda()
                idx_test = idx_test.cuda()
                idx_val = idx_val.cuda()
                if args.aggfunc == "maxpool":
                    row, col = row.cuda(), col.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            n_nodes = adj.shape[0]
            batch_size = args.batch_size
            num_batches = len(idx_train) // batch_size + 1

            state_dict_early_model = None
            best_val_acc = 0.0
            best_val_loss = 0.0

            for epoch in tqdm(range(args.epochs)):

                model.train()

                idx = list(range(len(idx_train)))
                np.random.shuffle(idx)

                for batch in range(num_batches):
                    optimizer.zero_grad()
                    cur_idx = idx_train[idx[batch * batch_size: batch * batch_size + batch_size]]
                    # For each batch, forward the whole graph but compute loss only on nodes in current batch
                    output = model(features, adj, cur_idx=cur_idx, verbose=False,row=row,col=col)
                    train_loss = F.nll_loss(output, labels[cur_idx])

                    train_acc = accuracy(output, labels[cur_idx])
                    train_loss.backward()
                    optimizer.step()

                # Validation for each epoch
                model.eval()
                with torch.no_grad():
                    output = model(features, adj, cur_idx=idx_val, verbose=False,row=row,col=col)

                    val_loss = F.nll_loss(output, labels[idx_val])
                    val_acc = accuracy(output, labels[idx_val])

                if args.verbose:
                    print(
                        "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}".format(
                            epoch, train_loss.item(), train_acc, val_loss, val_acc))

                if val_acc >= best_val_acc and (val_acc > best_val_acc or val_loss < best_val_loss):
                    best_val_acc = val_acc.cpu()
                    best_val_loss = val_loss.detach().cpu()
                    state_dict_early_model = deepcopy(model.state_dict())

            # Perform test
            with torch.no_grad():

                model.load_state_dict(state_dict_early_model)
                model.eval()

                output = model(features, adj, cur_idx=idx_test, verbose=True,row=row,col=col)
                acc_test = accuracy(output, labels[idx_test])

            final_acc.append(acc_test.detach().cpu().item())   
            print("Test_acc" + ":" + str(acc_test.detach().cpu().item()))

        final_acc = np.array(final_acc)
        print("Total accuracy: ", np.mean(final_acc) , " std: ", np.std(final_acc))
