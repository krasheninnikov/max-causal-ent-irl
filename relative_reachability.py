import numpy as np

def relative_reachability_penalty(mdp, horizon, start):
    '''
    Calculates the undiscounted relative reachability penalty for each state in an mdp, compared to the starting state baseline.

    Based on the algorithm described in: https://arxiv.org/pdf/1806.01186.pdf
    '''
    # TODO: Might want to get it to work with deterministic transitions
    mdp.get_transition_matrix()
    transition = np.reshape(mdp.T, (mdp.T.shape[1], mdp.T.shape[0], mdp.T.shape[2]))
    coverage = np.ones((mdp.nS, mdp.nS))
    for i in range(horizon):
        coverage = np.amax(transition * coverage, axis=0)
    
    return np.sum(np.maximum(coverage - coverage[start, :], 0), axis=1)
    
    # TODO: Clean up comments when possible. 
    # for i in range(mdp.nS):
    #     np.sum(np.maximum(coverage[i, :] - coverage[s, :])

    # last_c = np.ones(mdp.nS, mdp.nS)
    # for i in range(horizon):
    #   dp.nA:
    #       coverage = np.maximum(coverage, mdp.T[:, a, :] * last_c)
    #     last_c = coverage
    #  mdp.T
    
    #
    # For all s, s', compute C(s; s'). This can probably be done with some kind of Bellman equation
    # For all s, compute d(s, s_start) = \sum s' max(C(s_start, s')-C(s, s')). Can probably be done with a simple numpy operation once you have C(s; s')
