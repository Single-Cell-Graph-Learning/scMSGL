import numpy as np

from numba import njit

from . import rowsum_mat

@njit
def _update_auxiliaries(y_pos, y_neg):
    # Projection onto complementarity set
    n_pairs = len(y_pos)
    v = np.zeros((n_pairs, 1))
    w = np.zeros((n_pairs, 1))
    for i in range(n_pairs):
        if y_pos[i] < 0 and y_neg[i] < 0:
            if y_pos[i] < y_neg[i]:
                v[i] = y_pos[i]
            else:
                w[i] = y_neg[i]
        elif y_pos[i] < 0 and y_neg[i] >= 0:
            v[i] = y_pos[i]
        elif y_pos[i] >= 0 and y_neg[i] < 0:
            w[i] = y_neg[i]

    return v, w

def _update_view_laplacians(y, S, alpha, beta, rho, n_nodes):
    n_pairs = n_nodes*(n_nodes - 1)/2

    beta2 = 2*alpha + rho + 2*beta
    rho1 = alpha
    a = 1/beta2
    b = (4*rho1**2)/(beta2*(beta2 + rho1*(n_nodes-2))*(beta2 + 2*rho1*(n_nodes-1)))
    c = - rho1/(beta2*(beta2 + rho1*(n_nodes-2)))

    y = a*y + b*np.sum(y) + c*(S.T@(S@y))

    return y - (n_nodes + np.sum(y))/n_pairs

def _objective(data_vecs, l_views, l_consensus, alpha, beta, gamma, S):
    n_views = len(data_vecs)

    result = 0
    sign = {"+": 1, "-": -1} # for calculating smoothness
    for s in ["+", "-"]:
        for view in range(n_views):
            result += sign[s]*(data_vecs[view]).T@l_views[s][view] # smoothness
            result += 0.5*alpha*np.linalg.norm(S@l_views[s][view])**2 # degree term
            result += alpha*np.linalg.norm(l_views[s][view])**2 # sparsity term
            result += beta*np.linalg.norm(l_views[s][view] - l_consensus[s])**2 # consensus term
    
        result += gamma*np.linalg.norm(l_consensus[s], ord=1) # sparsity term for consensus

    return result.item()

def _signed_graphs(data_vecs, S, alpha, beta, gamma, rho=1, max_iter=1000):

    n_views = len(data_vecs)
    n_pairs = len(data_vecs[0]) # number of node pairs
    n_nodes = (1 + np.sqrt(8*n_pairs + 1))//2 # number of nodes
    
    # Initialization
    v_views = {"+": [None]*n_views, "-": [None]*n_views}
    l_views = {"+": [], "-": []}
    lambda_views = {"+": [], "-": []}
    for s in ["+", "-"]:
        for view in range(n_views):
            l_views[s].append(np.zeros((n_pairs, 1)))
            lambda_views[s].append(np.zeros((n_pairs, 1)))

    v_consensus = {"+": None, "-": None}
    l_consensus = {"+": np.zeros((n_pairs, 1)), "-": np.zeros((n_pairs, 1))}
    lambda_consensus = {"+": np.zeros((n_pairs, 1)), "-": np.zeros((n_pairs, 1))}
    
    # ADMM Iterations
    objective_vals = []
    for iter in range(max_iter):
        
        # Update auxiliary variables
        for view in range(n_views):
            y_pos = l_views["+"][view] - lambda_views["+"][view]/rho
            y_neg = l_views["-"][view] - lambda_views["-"][view]/rho
            v_views["+"][view], v_views["-"][view] = _update_auxiliaries(y_pos, y_neg)

        y_pos = l_consensus["+"] - lambda_consensus["+"]/rho
        y_neg = l_consensus["-"] - lambda_consensus["-"]/rho
        v_consensus["+"], v_consensus["-"] = _update_auxiliaries(y_pos, y_neg)

        # Update view Laplacians
        for view in range(n_views):
            y = 2*beta*l_consensus["+"] + lambda_views["+"][view] + rho*v_views["+"][view] - data_vecs[view]
            l_views["+"][view] = _update_view_laplacians(y, S, alpha, beta, rho, n_nodes)

        for view in range(n_views):
            y = 2*beta*l_consensus["-"] + lambda_views["-"][view] + rho*v_views["-"][view] + data_vecs[view]
            l_views["-"][view] = _update_view_laplacians(y, S, alpha, beta, rho, n_nodes)

        # Update consensus Laplacians
        for s in ["+", "-"]:
            y = np.zeros((n_pairs, 1))
            for view in range(n_views):
                y += 2*beta*l_views[s][view]
            y += lambda_consensus[s] + rho*v_consensus[s]
            y /= (2*beta*n_views + rho)
            threshold = gamma/(2*beta*n_views + rho)
            l_consensus[s] = np.maximum(0, y-threshold) + np.minimum(0, y+threshold)

        # Update Lagrangian multipliers
        for s in ["+", "-"]:
            for view in range(n_views):
                lambda_views[s][view] += rho*(v_views[s][view] - l_views[s][view])
            lambda_consensus[s] += rho*(v_consensus[s] - l_consensus[s])

        # Calculate objective
        # TODO: Calculate Augmented Lagrangian
        objective_vals.append(_objective(data_vecs, v_views, v_consensus, alpha, beta, gamma, S))

        if iter > 10 and abs(objective_vals[-1] - objective_vals[-2]) < 1e-4:
            break

    # Remove small edges and convert to the adjacency matrix
    for s in ["+", "-"]:
        for view in range(n_views):
            v_views[s][view][v_views[s][view]>-1e-4] = 0
            v_views[s][view] = np.abs(v_views[s][view])
            
        v_consensus[s][v_consensus[s]>-1e-4] = 0
        v_consensus[s] = np.abs(v_consensus[s])

    return v_views, v_consensus, objective_vals

def _view_correlations(w_views):
    n_views = len(w_views)
    mean_corr = 0
    for vi in range(n_views):
        for vj in range(vi+1, n_views):
            mean_corr += np.corrcoef(np.squeeze(w_views[vi]), 
                                      np.squeeze(w_views[vj]))[0,1]
    mean_corr /= n_views*(n_views-1)/2
    return mean_corr

def _density(w):
    return np.count_nonzero(w)/len(w)

def learn_multiple_signed_graphs(X, assoc_func, density, relation, **kwargs):
    # Input check
    if not isinstance(X, list):
        raise Exception("Mutliple sets of graph signals must be provided when ", +\
                        "learning multiple signed graphs.")

    n_views = len(X)
    n_nodes, n_signals = X[0].shape
    S = rowsum_mat(n_nodes)

    # Data preparation: Get 2k-S^Td for each view
    data_vecs = []
    for v in range(n_views):
        k, d = assoc_func(X[v])
        data_vecs.append(2*k - S.T@d)
        if np.ndim(data_vecs[v]) == 1:
            data_vecs[v] = data_vecs[v][:, None]

    alpha = kwargs["alpha"] if "alpha" in kwargs else 1
    beta = kwargs["beta"] if "beta" in kwargs else 1
    gamma = kwargs["gamma"] if "gamma" in kwargs else 0.1
    rho = kwargs["rho"] if "rho" in kwargs else 10
    max_iter = kwargs["max_iter"] if "max_iter" in kwargs else 100

    if relation == 0:
        beta = 0
        gamma = 0

    rng = np.random.default_rng()

    iter = 0
    while True:
        iter += 1

        # print(alpha, beta, gamma)

        w_views, w_consensus, _ = _signed_graphs(
            data_vecs, S, alpha, beta, gamma, rho, max_iter
        )

        # Calculate densities
        obtained_relation = 0
        for s in ["+", "-"]:
            obtained_relation += _view_correlations(w_views[s])/2

        # Calculate view correlations
        view_density = 0
        consensus_density = 0
        for s in ["+", "-"]:
            for v in range(n_views):
                view_density += _density(w_views[s][v])/(2*(n_views))
            consensus_density += _density(w_consensus[s])/2

        # Update parameters
        update_alpha = abs(view_density - density) > 1e-2
        update_beta = abs(obtained_relation - relation) > 1e-2 if relation > 0 else False
        update_gamma = abs(consensus_density - density) > 1e-2 if relation > 0 else False

        if update_beta:
            rnd = rng.uniform(0.8, 1.0)
            diff = obtained_relation - relation
            if diff > 0:
                beta = max(beta - 2*abs(diff), beta*(1-rnd*0.3)) 
            else:
                beta = min(beta + 2*abs(diff), beta*(1+rnd*0.3))

        if update_alpha:
            rnd = rng.uniform(0.8, 1.0)
            diff = view_density - density
            if diff > 0:
                alpha = max(alpha - 2*abs(diff), alpha*(1-rnd*0.3)) 
            else:
                alpha = min(alpha + 2*abs(diff), alpha*(1+rnd*0.3))

        if update_gamma:
            rnd = rng.uniform(0.8, 1.0)
            diff = consensus_density - density
            if diff > 0:
                gamma = min(gamma + 2*abs(diff), gamma*(1+rnd*0.3)) 
            else:
                gamma = max(gamma - 2*abs(diff), gamma*(1-rnd*0.3))

        if (not update_beta) and (not update_alpha) and (not update_gamma):
            break

        if iter > 50:
            print("Hyperparameter search run too much. It is aborted.",
                  "Try initiating them at different values.")
            break

    # Optimal parameter values
    params = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma
    }

    return w_views, w_consensus, params