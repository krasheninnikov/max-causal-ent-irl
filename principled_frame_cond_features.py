import numpy as np
from scipy.stats import norm, laplace

from value_iter_and_policy import vi_boltzmann, vi_boltzmann_deterministic
from envs.utils import unique_perm, zeros_with_ones, printoptions
from occupancy_measure import compute_d_deterministic, compute_d_last_step_deterministic

def grad_gaussian_prior(theta, theta_spec, sigma=1):
    return -(theta-theta_spec)/(sigma**2)


def grad_laplace_prior(theta, theta_spec, b=1):
    return (theta_spec-theta)/(np.fabs(theta-theta_spec)*b)


class norm_distr(object):
    def __init__(self, mu, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.distribution = norm(loc=mu, scale=sigma)
        
    def rvs(self):
        '''sample'''
        return self.distribution.rvs()
    
    def pdf(self, x):
        return self.distribution.pdf(x)
    
    def logpdf(self, x):
        return self.distribution.logpdf(x)
    
    def logdistr_grad(self, x):
        return (self.mu-x)/(self.sigma**2)
    

class laplace_distr(object):
    def __init__(self, mu, b=1):
        self.mu = mu
        self.b = b
        self.distribution = laplace(loc=mu, scale=b)
        
    def rvs(self):
        '''sample'''
        return self.distribution.rvs()
    
    def pdf(self, x):
        return self.distribution.pdf(x)
    
    def logpdf(self, x):
        return self.distribution.logpdf(x)
    
    def logdistr_grad(self, x):
        return (self.mu-x)/(np.fabs(x-self.mu)*self.b)


def compute_g_deterministic(mdp, policy, p_0, T, d_last_step_list, feature_matrix):
    # base case
    G_prev = np.zeros((mdp.nS, feature_matrix.shape[1]))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            # due to env being deterministic, sprime=self.P[s][a][0][1] and p_sprime=1
            G_prev[mdp.P[s][a][0][1], :] += p_0[s] * policy[s, a] * feature_matrix[s,:]

    # recursive case
    for t in range(T-1):
        G = np.zeros((mdp.nS, feature_matrix.shape[1]))
        for s in range(mdp.nS):
            for a in range(mdp.nA):
                # due to env being deterministic, sprime=self.P[s][a][0][1] and p_sprime=1
                G[mdp.P[s][a][0][1], :] += policy[s, a] * (G_prev[s] + d_last_step_list[t][s] * feature_matrix[s,:]) 

        G_prev = np.copy(G)
    return G


def om_method(mdp, s_current, p_0, horizon, temp=1, epochs=1, learning_rate=0.2, r_prior=None, r_vec=None):
    '''Modified MaxCausalEnt that maximizes last step occupancy measure for the current state'''
    if r_vec is None:
        r_vec = .01*np.random.randn(mdp.f_matrix.shape[1])
    print('Initial reward vector: {}'.format(r_vec))

    for i in range(epochs):
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s)
        V, Q, policy = vi_boltzmann_deterministic(mdp, 1, mdp.f_matrix @ r_vec, horizon, temp)

        # Compute the gradient
        d_last_step, d_last_step_list = compute_d_last_step_deterministic(mdp, policy, p_0, horizon, return_all=True)
        G = compute_g_deterministic(mdp, policy, p_0, horizon, d_last_step_list, mdp.f_matrix)
        d_T_step = compute_d_deterministic(mdp, 1, policy, p_0, horizon+1)

        g_div_d_last_step = np.zeros(mdp.f_matrix.shape[1])
        if d_last_step[np.where(s_current)]!=0:
            g_div_d_last_step = G[np.where(s_current)]/d_last_step[np.where(s_current)]

        dL_dr_vec = g_div_d_last_step.flatten() + (s_current - d_T_step) @ mdp.f_matrix

        # Gradient of the prior
        if r_prior!= None: dL_dr_vec += r_prior.logdistr_grad(r_vec)

        # Gradient ascent
        r_vec = r_vec + learning_rate * dL_dr_vec

        if i%1==0:
            with printoptions(precision=4, suppress=True):
                print('Epoch {}; Reward vector: {}'.format(i, r_vec))
    return r_vec
