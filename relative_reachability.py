import numpy as np

def relative_reachability_penalty(mdp, horizon, start):
    '''
    Calculates the undiscounted relative reachability penalty for each state in an mdp, compared to the starting state baseline. 
     
    Based on the algorithm described in: https://arxiv.org/pdf/1806.01186.pdf
    '''
    coverage = get_coverage(mdp, horizon)
    distributions = baseline_state_distributions(mdp, horizon, start)

    def penalty(state):
        return np.sum(np.maximum(coverage[state, :] - coverage, 0), axis=1)

    def penalty_for_baseline_distribution(dist):
        return sum((dist[state] * penalty(state) for state in range(mdp.nS) if dist[state] != 0))

    r_r = np.array(list(map(penalty_for_baseline_distribution, distributions)))
    if np.amax(r_r) == 0:
        return np.zeros_like(r_r)
    return r_r / np.amax(r_r)

def get_coverage(mdp, horizon):
    coverage = np.identity(mdp.nS)
    for i in range(horizon):
        # coverage(s0, sk) = \max_{a0} \sum_{s1} P(s1 | s0, a) * coverage(s1, sk)
        action_coverage = mdp.T_matrix.dot(coverage)
        action_coverage = action_coverage.reshape((mdp.nS, mdp.nA, mdp.nS))
        coverage = np.amax(action_coverage, axis=1)
    return coverage

def baseline_state_distributions(mdp, horizon, start):
    distribution = np.zeros(mdp.nS)
    distribution[start] = 1
    distributions = [ distribution ]
    for _ in range(horizon - 1):
        distribution = mdp.baseline_matrix_transpose.dot(distribution)
        distributions.append(distribution)
    return distributions
