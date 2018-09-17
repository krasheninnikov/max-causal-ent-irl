import numpy as np

def relative_reachability_penalty(mdp, horizon, start):
    '''
    Calculates the undiscounted relative reachability penalty for each state in an mdp, compared to the starting state baseline.

    Based on the algorithm described in: https://arxiv.org/pdf/1806.01186.pdf
    '''
    coverage = np.identity(mdp.nS)
    for i in range(horizon):
        # print(i)
        coverage = np.maximum.reduce([coverage[mdp.deterministic_T[:, a], :] for a in range(mdp.nA)])
    
    return np.sum(np.maximum(coverage[mdp.get_num_from_state(start), :] - coverage, 0), axis=1)