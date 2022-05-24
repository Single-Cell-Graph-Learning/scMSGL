from turtle import pos
import numpy as np

from numba import njit

from . import rowsum_mat

@njit
def _vw_step(yv, yw):
    M = len(yv)
    v = np.zeros((M, 1))
    w = np.zeros((M, 1))
    for i in range(M):
        if yv[i] < 0 and yw[i] < 0:
            if yv[i] < yw[i]:
                v[i] = yv[i]
            else:
                w[i] = yw[i]
        elif yv[i] < 0 and yw[i] >= 0:
            v[i] = yv[i]
        elif yv[i] >= 0 and yw[i] < 0:
            w[i] = yw[i]

    return v, w

def _l_step(y, P, alpha, rho, n, m):
    beta2 = 4*alpha + rho
    rho1 = 2*alpha
    a = 1/beta2
    b = (4*rho1**2)/(beta2*(beta2 + rho1*(n-2))*(beta2 + 2*rho1*(n-1)))
    c = - rho1/(beta2*(beta2 + rho1*(n-2)))

    y = a*y + b*np.sum(y) + c*(P.T@(P@y))

    return y - (n + np.sum(y))/m

def _objective(z, lpos, lneg, alpha1, alpha2, P):
    result = 0
    result += (z.T@lpos).item() 
    result -= (z.T@lneg).item() 
    result += 2*alpha1*np.linalg.norm(lpos)**2
    result += alpha1*np.linalg.norm(P@lpos)**2
    result += 2*alpha2*np.linalg.norm(lneg)**2
    result += alpha2*np.linalg.norm(P@lneg)**2

    return result

def _signed_graph(z, P, alpha1, alpha2, rho=10, max_iter=10000):
    # TODO: Docstring and clean
    # TODO: What is optimal rho

    rng = np.random.default_rng()

    M = len(z) # number of node pairs
    n = (1 + np.sqrt(8*M + 1))//2  # number of nodes

    # Initialization
    lpos = np.zeros((M, 1))
    lneg = np.zeros((M, 1))
    lambda_pos = np.zeros((M, 1))
    lambda_neg = np.zeros((M, 1))

    objective_vals = []

    # ADMM iterations
    for iter in range(max_iter):

        # v, w steps
        yv = lpos - lambda_pos/rho
        yw = lneg - lambda_neg/rho
        v, w = _vw_step(yv, yw)

        # positive l step
        if alpha1 > 0:
            y = rho*v + lambda_pos - z
            lpos = _l_step(y, P, alpha1, rho, n, M)

        # negative l step
        if alpha2 > 0:
            y = rho*w + lambda_neg + z
            lneg = _l_step(y, P, alpha2, rho, n, M)

        # multipliers update
        lambda_pos += rho*(v - lpos)
        lambda_neg += rho*(w - lneg)

        # Calculate objective
        objective_vals.append(_objective(z, v, w, alpha1, alpha2, P))

        # Convergence
        if iter > 10 and abs(objective_vals[-1] - objective_vals[-2]) < 1e-4:
            break

    v[v>-1e-4] = 0
    w[w>-1e-4] = 0
    v = np.abs(v)
    w = np.abs(w)
    return v, w, objective_vals

def _density(w):
    return np.count_nonzero(w)/len(w)

def learn_a_signed_graph(X, assoc_func, density_pos, density_neg, **kwargs):
    
    n_nodes, n_signals = X.shape
    S = rowsum_mat(n_nodes)

    # Data preparation: Get 2k-S^Td
    k, d = assoc_func(X)
    data_vec = 2*k - S.T@d
    if np.ndim(data_vec) == 1:
        data_vec = data_vec[:, None]

    alpha_pos = kwargs["alpha_pos"] if "alpha_pos" in kwargs else 1
    alpha_neg = kwargs["alpha_neg"] if "alpha_neg" in kwargs else 1
    rho = kwargs["rho"] if "rho" in kwargs else 10
    max_iter = kwargs["max_iter"] if "max_iter" in kwargs else 100

    if density_pos == 0:
        alpha_pos = 0
    if density_neg == 0:
        alpha_neg = 0
    
    iter = 0
    while True:
        iter += 1

        # Learn signed graph
        w_pos, w_neg, _ = _signed_graph(
            data_vec, S, alpha_pos, alpha_neg, rho, max_iter
        )

        # Calculate densities
        obtained_density_pos = _density(w_pos)
        obtained_density_neg = _density(w_neg)

        # Update parameters
        update_alpha_pos = abs(obtained_density_pos - density_pos) > 1e-2
        update_alpha_neg = abs(obtained_density_neg - density_neg) > 1e-2

        if density_pos == 0:
            update_alpha_pos = False
        if density_neg == 0:
            update_alpha_neg = False

        if update_alpha_pos:
            diff = obtained_density_pos - density_pos
            if diff > 0:
                alpha_pos = max(alpha_pos - 2*abs(diff), alpha_pos*0.7) 
            else:
                alpha_pos = min(alpha_pos + 2*abs(diff), alpha_pos*1.3)

        if update_alpha_neg:
            diff = obtained_density_neg - density_neg
            if diff > 0:
                alpha_neg = max(alpha_neg - 2*abs(diff), alpha_pos*0.7) 
            else:
                alpha_neg = min(alpha_neg + 2*abs(diff), alpha_neg*1.3)

        if (not update_alpha_pos) and (not update_alpha_neg):
            break

    # Optimal parameter values
    params = {
        "alpha_pos": alpha_pos,
        "alpha_neg": alpha_neg
    }

    return w_pos, w_neg, params
