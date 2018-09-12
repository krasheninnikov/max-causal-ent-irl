import numpy as np
from value_iter_and_policy import vi_boltzmann, vi_boltzmann_deterministic
from envs.utils import unique_perm, zeros_with_ones, printoptions

def compute_d_last_step_discrete(mdp, policy, p_0, T, verbose=False):
    '''Computes the last-step occupancy measure'''
    
    D_prev = p_0 
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
    return D

def om_method(mdp, s_current, p_0, horizon, temp=1, epochs=1, learning_rate=0.2, r_vec=None):
    '''Modified MaxCausalEnt that maximizes last step occupancy measure for the current state'''
     
    if r_vec is None:
        r_vec = .01*np.random.randn(mdp.f_matrix.shape[1])
        print('Initial reward vector: {}'.format(r_vec))
        
    for i in range(epochs):
        
            # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s) 
            V, Q, policy = vi_boltzmann_deterministic(mdp, 1, mdp.f_matrix @ r_vec, horizon, temp) 
            
            D = compute_d_last_step_discrete(mdp, policy, p_0, horizon)   
            dL_dr_vec = -(s_current - D) @ mdp.f_matrix

            # Gradient descent; gradient may not be the actual gradient -- have to check the math,
            # bit this should perform the matching correctly
            r_vec = r_vec - learning_rate * dL_dr_vec
            
            if i%40==0:
                with printoptions(precision=4, suppress=True):
                    print('Epoch {}; Reward vector: {}'.format(i, r_vec))

    return r_vec
