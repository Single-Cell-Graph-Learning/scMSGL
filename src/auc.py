from itertools import combinations, permutations
from operator import itemgetter

import pandas as pd
import numpy as np

from scipy import sparse
from sklearn.metrics import average_precision_score as auprc

def signed(true_edges, pred_edges, type) -> pd.DataFrame:
    is_pred_signed = np.count_nonzero(pred_edges.EdgeWeight >= 0) != len(pred_edges)

    nodes = {
        g: i for i, g in enumerate(np.unique(true_edges.loc[:, ["Gene1", "Gene2"]]))
    }
    n_nodes = len(nodes)
    
    if not is_pred_signed:
        ignored_edges = {
            "+": set([(nodes[g1], nodes[g2]) for g1, g2, s in 
                    true_edges.itertuples(index=False, name=None) if s == "-"]),
            "-": set([(nodes[g1], nodes[g2]) for g1, g2, s in 
                    true_edges.itertuples(index=False, name=None) if s == "+"])
        }

    auprc_ratios = {}
    for sgn in ["+", "-"]:

        edges = np.array(
            [[1, nodes[g1], nodes[g2]] for g1, g2, s in 
             true_edges.itertuples(index=False, name=None) if s == sgn and g1 != g2]
        )
        true_adj = sparse.csr_matrix((edges[:, 0], (edges[:, 1], edges[:, 2])), 
                                     shape=(n_nodes, n_nodes))

        pred_edge_sign = lambda w: "+" if w>0 else "-"
        if is_pred_signed:
            edges = np.array(
                [[np.abs(ew), nodes[g1], nodes[g2]] 
                 for g1, g2, ew in pred_edges.itertuples(index=False, name=None) 
                 if ew != 0 and pred_edge_sign(ew) == sgn and g1 != g2]
            )
        else:
            edges = np.array(
                [[np.abs(ew), nodes[g1], nodes[g2]]
                 for g1, g2, ew in pred_edges.itertuples(index=False, name=None) 
                 if (ew != 0) and ((nodes[g1], nodes[g2]) not in ignored_edges[sgn])
                    and g1 != g2]
            )

        if len(edges) == 0:
            auprc_ratios[sgn] = 1
            continue

        pred_adj = sparse.csr_matrix((edges[:, 0], (edges[:, 1], edges[:, 2])), 
                                     shape=(n_nodes, n_nodes))

        if type == "undirected":
            true_adj += true_adj.T
            pred_adj += pred_adj.T

            # Scikit-learn AUPRC function does not accept sparse matrices ??
            true_adj = true_adj.toarray()[np.triu_indices(n_nodes, k=1)]
            pred_adj = pred_adj.toarray()[np.triu_indices(n_nodes, k=1)]

        else:
            true_adj = true_adj.toarray()[~np.eye(n_nodes, dtype=bool)]
            pred_adj = pred_adj.toarray()[~np.eye(n_nodes, dtype=bool)]

        auprc_sgn = auprc(true_adj, pred_adj)
        auprc_rnd = np.count_nonzero(true_adj)/len(true_adj)
        auprc_ratios[sgn] = auprc_sgn/auprc_rnd

    return auprc_ratios