import numpy as np


def compute_D(mdp, gamma, policy, P_0=None, t_max=None, threshold=1e-6):
    '''
    Computes occupancy measure of a MDP under a given time-constrained policy
    -- the expected discounted number of times that policy π visits state s in
    a given number of timesteps.

    The version w/o discount is described in Algorithm 9.3 of Ziebart's thesis:
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.

    The discounted version can be found in the supplement to Levine's
    "Nonlinear Inverse Reinforcement Learning with Gaussian Processes" (GPIRL):
    https://graphics.stanford.edu/projects/gpirl/gpirl_supplement.pdf.

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    gamma : float
        Discount factor; 0<=gamma<=1.
    policy : 2D numpy array
        policy[s,a] is the probability of taking action a in state s.
    P_0 : 1D numpy array of shape (mdp.nS)
        i-th element is the probability that the traj will start in state i.
    t_max : int
        number of timesteps the policy is executed.

    Returns
    -------
    1D numpy array of shape (mdp.nS)
    '''

    if P_0 is None: P_0 = np.ones(mdp.nS) / mdp.nS
    D_prev = np.zeros_like(P_0)

    t = 0
    diff = float("inf")
    while diff > threshold:

        # ∀ s: D[s] <- P_0[s]
        D = np.copy(P_0)

        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # for all s_prime reachable from s by taking a do:
                for p_sprime, s_prime, _ in mdp.P[s][a]:
                    D[s_prime] += gamma * D_prev[s] * policy[s, a] * p_sprime

        diff = np.amax(abs(D_prev - D))
        D_prev = np.copy(D)

        if t_max is not None:
            t+=1
            if t==t_max: break

    return D


def compute_d_deterministic(mdp, gamma, policy, P_0=None, t_max=None, threshold=1e-6):
    '''
    Same as compute_D, but works only for deterministic dynamics and is faster
    '''

    if P_0 is None: P_0 = np.ones(mdp.nS) / mdp.nS
    D_prev = np.zeros_like(P_0)

    t = 0
    diff = float("inf")
    while diff > threshold:

        # ∀ s: D[s] <- P_0[s]
        D = np.copy(P_0)

        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # for all s_prime reachable from s by taking a do:
                D[mdp.P[s][a][0][1]] += gamma* D_prev[s] * policy[s, a]

        diff = np.amax(abs(D_prev - D))
        D_prev = np.copy(D)

        if t_max is not None:
            t+=1
            if t==t_max: break

    return D


def compute_d_last_step_deterministic(mdp, policy, p_0, T, verbose=False, return_all=False):
    '''Computes the last-step occupancy measure'''
    D_prev = p_0
    d_last_step_list = [D_prev]

    t = 0
    for t in range(T):

        # for T-step OM we'd do D=np.copy(P_0). However, we want the last step one, so:
        D = np.zeros_like(p_0)

        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # due to env being deterministic, sprime=self.P[s][a][0][1] and p_sprime=1
                D[mdp.P[s][a][0][1]] += D_prev[s] * policy[s, a]

        D_prev = np.copy(D)
        if verbose is True: print(D)
        if return_all: d_last_step_list.append(D)

    if return_all:
        return D, d_last_step_list
    else:
        return D
