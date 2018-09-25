import numpy as np
from value_iter_and_policy import softmax

def value_iter(mdp, gamma, r, horizon, temperature=1, threshold=1e-10, time_dependent_reward=False):
        '''
        Finds the optimal state and state-action value functions via value
        iteration with the "soft" max-ent Bellman backup:

        Q_{sa} = r_s + gamma * \sum_{s'} p(s'|s,a)V_{s'}
        V'_s = temperature * log(\sum_a exp(Q_{sa}/temperature))

        Computes the Boltzmann rational policy
        \pi_{s,a} = exp((Q_{s,a} - V_s)/temperature).

        Parameters
        ----------
        mdp : object
            Instance of the Env class (see envs/env.py).

        gamma : float
            Discount factor; 0<=gamma<=1.
        r : 1D numpy array
            Initial reward vector with the length equal to the
            number of states in the MDP.
        horizon : int
            Horizon for the finite horizon version of value iteration.
        temperature: float
            Rationality constant to use in the value iteration equation.
        threshold : float
            Convergence threshold.

        Returns
        -------
        1D numpy array
            Array of shape (mdp.nS, 1), each V[s] is the value of state s under
            the reward r and Boltzmann policy.
        2D numpy array
            Array of shape (mdp.nS, mdp.nA), each Q[s,a] is the value of
            state-action pair [s,a] under the reward r and Boltzmann policy.
        List of 2D numpy arrays
            Arrays of shape (mdp.nS, mdp.nA), each value p[t][s,a] is the
            probability of taking action a in state s at time t.
        '''
        nS, nA = mdp.nS, mdp.nA
        # Functions for computing the policy
        expt = lambda x: np.exp(x/temperature)
        tlog = lambda x: temperature * np.log(x)

        if not time_dependent_reward:
            r = [r] * horizon  # Fast, since we aren't making copies

        policies = []
        V = np.copy(r[horizon-1])
        for t in range(horizon-2, -1, -1):
            future_values = mdp.T_matrix.dot(V).reshape((nS, nA))
            Q = np.expand_dims(r[t], axis=1) + gamma * future_values

            if temperature is None:
                V = Q.max(axis=1)
                # Argmax to find the action number, then index into np.eye to
                # one hot encode. Note this will deterministically break ties
                # towards the smaller action.
                policy = np.eye(nA)[np.argmax(Q, axis=1)]
            else:
                # ∀ s: V_s = temperature * log(\sum_a exp(Q_sa/temperature))
                # ∀ s,a: policy_{s,a} = exp((Q_{s,a} - V_s)/t)
                V = softmax(Q, temperature)
                policy = expt(Q - np.expand_dims(V, axis=1))

            policies.append(policy)

        return policies[::-1]


def evaluate_policy(mdp, policy, start, gamma, r, horizon):
    """Expected reward from the policy."""
    V = r
    for t in range(horizon-2, -1, -1):
        future_values = mdp.T_matrix.dot(V).reshape((mdp.nS, mdp.nA))
        Q = np.expand_dims(r, axis=1) + gamma * future_values
        V = np.sum(policy[t] * Q, axis=1)
    return V[start]
