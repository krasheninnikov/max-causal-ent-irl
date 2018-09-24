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
    
    r_r = np.sum(np.maximum(coverage[mdp.get_num_from_state(start), :] - coverage, 0), axis=1)
    return  r_r / np.amax(r_r) 

def stochastic_relative_reachability_penalty(mdp, horizon, start):
    '''
    Calculates the undiscounted relative reachability penalty for each state in an mdp, compared to the starting state baseline. 
     
    Based on the algorithm described in: https://arxiv.org/pdf/1806.01186.pdf
    '''
    # mdp.get_transition_matrix()
    mdp.T = np.zeros((mdp.nS, mdp.nA, mdp.nS))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            mdp.T[s, a, mdp.deterministic_T[s, a]] = 1

    transition = np.reshape(mdp.T, (mdp.T.shape[1], mdp.T.shape[0], mdp.T.shape[2]))
    coverage = np.identity(mdp.nS)
    for i in range(horizon):
        coverage = np.amax(transition * coverage, axis=0)
    
    r_r = np.sum(np.maximum(coverage[mdp.get_num_from_state(start), :] - coverage, 0), axis=1)
    #TODO: Consider nans
    return r_r / np.amax(r_r)