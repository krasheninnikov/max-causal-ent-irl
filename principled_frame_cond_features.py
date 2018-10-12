import numpy as np
from scipy.stats import norm, laplace
from scipy.optimize import check_grad, approx_fprime

from value_iter import value_iter
from envs.utils import unique_perm, zeros_with_ones, printoptions


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


def compute_d_last_step_parallel(mdp, policy, p_0, T, gamma=1, verbose=False, return_all=False):
    '''Computes the last-step occupancy measure'''
    if len(p_0.shape)==1:
        p_0 = p_0.reshape((-1,1))
    n = p_0.shape[0]

    d_last_step_array = np.zeros((T, p_0.shape[0], p_0.shape[1]))
    D, d_last_step_array[0, :, :] = p_0, p_0
    i=1
    for t in range(T-1):
        # D(s') = \sum_{s, a} D_prev(s) * p(a | s) * p(s' | s, a)
        state_action_prob = (np.expand_dims(D, axis=2) * policy[t])
        D = mdp.T_matrix_transpose.dot(state_action_prob.reshape(n,-1).T)
        D = D.T

        if return_all: d_last_step_array[i]=D
        i+=1

        if verbose is True: print(D)

    return (D, d_last_step_array) if return_all else D


def compute_f_matmul(mdp, policy, p_0, s_current, horizon):
    '''second term of the RLSP gradient'''
    def ones_at_nonzero(x):
        '''Returns a matrix of the shape [n,len(x)], where n is the number of non-zero elements in x'''
        n = len(np.nonzero(x)[0])
        out = np.zeros((n, x.shape[0]))
        out[np.arange(n), np.nonzero(x)[0]] =1
        return out

    F = np.zeros(mdp.f_matrix.shape[1], dtype='float64')

    d_last_step, d_last_step_list = compute_d_last_step_parallel(mdp, policy, ones_at_nonzero(p_0), horizon, return_all=True)

    feature_expectations = np.sum(d_last_step_list, axis=0).reshape((-1, mdp.nS)) @ mdp.f_matrix
    w = (ones_at_nonzero(p_0) @ p_0) * d_last_step[:, s_current]
    F += (w.reshape(1,-1) @ feature_expectations).flatten()
    return F


def compute_g(mdp, policy, p_0, T, d_last_step_list):
    # base case
    G = np.expand_dims(p_0, axis=1) * mdp.f_matrix

    d_last_step, d_last_step_array = compute_d_last_step_parallel(mdp, policy, np.eye(mdp.nS), T, return_all=True)
    print((d_last_step_array @ mdp.f_matrix).shape, 'd_last_step_array.shape')

    # recursive case
    for t in range(T-1):
        # G(s') = \sum_{s, a} p(a | s) p(s' | s, a) [ d_last_step_list[t] feature_matrix[s'] + G_prev[s] ]
        # Distribute the addition to get two different terms
        G_first = np.expand_dims(d_last_step_list[t].reshape(mdp.nS), axis=1) * policy[t]
        G_first = G_first.reshape((mdp.nS * mdp.nA,))
        G_first = mdp.T_matrix_transpose.dot(G_first)
        G_first = np.expand_dims(G_first, axis=1) * mdp.f_matrix

        G_second = np.expand_dims(policy[t], axis=-1) * np.expand_dims(G, axis=1)
        G_second = G_second.reshape((mdp.nS * mdp.nA, mdp.num_features))
        G_second = mdp.T_matrix_transpose.dot(G_second)

        G_corr = np.zeros_like(G_second)
        E_f = np.sum(d_last_step_array[0:T-t-1, :,:], axis = 0) @ mdp.f_matrix
        for s_t_p_1 in range(mdp.nS):
            s_a_pos_index=0
            for s_t in range(mdp.nS):
                for a_t in range(mdp.nA):

                    one_hot_s_t_p_1 = np.zeros(mdp.nS)
                    one_hot_s_t_p_1[s_t_p_1] = 1
                    p_diff = mdp.T_matrix[s_a_pos_index, :] - one_hot_s_t_p_1

                    sum_comp = p_diff @ E_f
                    #print(sum_comp.shape, 'sum_comp.shape')

                    G_corr[s_t_p_1,:] += policy[t][s_t, a_t] * d_last_step_list[t][s_t] * mdp.T_matrix[s_a_pos_index, s_t_p_1] * sum_comp

                    s_a_pos_index +=1

        G = G_first + G_second + G_corr

    return G


def compute_d_last_step(mdp, policy, p_0, T, gamma=1, verbose=False, return_all=False):
    '''Computes the last-step occupancy measure'''
    D, d_last_step_list = p_0, [p_0]
    for t in range(T-1):
        # D(s') = \sum_{s, a} D_prev(s) * p(a | s) * p(s' | s, a)
        state_action_prob = np.expand_dims(D, axis=1) * policy[t]
        D = mdp.T_matrix_transpose.dot(state_action_prob.flatten())

        if verbose is True: print(D)
        if return_all: d_last_step_list.append(D)

    return (D, d_last_step_list) if return_all else D


def compute_f(mdp, policy, p_0, s_current, horizon):
    '''second term of the RLSP gradient'''
    F = np.zeros(mdp.f_matrix.shape[1], dtype='float64')
    # TODO make a function to compute d_last_step_list for many different s_0
    # in parallel, and replace the for loop below with matrix multiplication
    for s_0 in np.nonzero(p_0)[0]:
        s_0_vec = np.zeros(mdp.nS)
        s_0_vec[s_0]=1

        d_last_step, d_last_step_list = compute_d_last_step(mdp, policy, s_0_vec, horizon, return_all=True)
        if d_last_step[s_current] == 0:
            continue

        feature_expectations = sum(d_last_step_list) @ mdp.f_matrix

        F += p_0[s_0] * d_last_step[s_current] * feature_expectations.reshape((mdp.f_matrix.shape[1]))
    return F


def om_method(mdp, s_current, p_0, horizon, temp=1, epochs=1, learning_rate=0.2,
              r_prior=None, r_vec=None, threshold=1e-3, check_grad_flag=True):
    '''The RLSP algorithm'''
    # p_0 = np.zeros(mdp.nS)
    # p_0[s_current]=.5
    # p_0[s_current+1]=.5
    def compute_grad_new(r_vec):
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s)
        policy = value_iter(mdp, 1, mdp.f_matrix @ r_vec, horizon, temp)

        d_last_step, d_last_step_list = compute_d_last_step(
            mdp, policy, p_0, horizon, return_all=True)
        if d_last_step[s_current] == 0:
            print('Error in om_method: No feasible trajectories!')
            return r_vec

        G = compute_g(mdp, policy, p_0, horizon, d_last_step_list)
        F = compute_f(mdp, policy, p_0, s_current, horizon)

        # Compute the gradient
        dL_dr_vec = (G[s_current]  - F)/ d_last_step[s_current]
        # Gradient of the prior
        if r_prior!= None: dL_dr_vec += r_prior.logdistr_grad(r_vec)

        return dL_dr_vec

    def compute_grad(r_vec):
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s)
        policy = value_iter(mdp, 1, mdp.f_matrix @ r_vec, horizon, temp)

        d_last_step, d_last_step_list = compute_d_last_step(
            mdp, policy, p_0, horizon, return_all=True)
        if d_last_step[s_current] == 0:
            print('Error in om_method: No feasible trajectories!')
            return r_vec

        feature_expectations = sum(d_last_step_list) @ mdp.f_matrix

        G = compute_g(mdp, policy, p_0, horizon, d_last_step_list)
        # Compute the gradient
        dL_dr_vec = G[s_current] / d_last_step[s_current] - feature_expectations
        # Gradient of the prior
        if r_prior!= None: dL_dr_vec += r_prior.logdistr_grad(r_vec)

        return dL_dr_vec

    def compute_log_likelihood(r_vec):
        policy = value_iter(mdp, 1, mdp.f_matrix @ r_vec, horizon, temp)
        d_last_step, d_last_step_list = compute_d_last_step(
                    mdp, policy, p_0, horizon, return_all=True)
        log_likelihood = np.log(d_last_step[s_current])
        if r_prior!= None: log_likelihood += np.sum(r_prior.logpdf(r_vec))

        return log_likelihood

    def get_grad(_):
        "dummy function for use with check_grad()"
        return dL_dr_vec

    if r_vec is None:
        r_vec = 0.01*np.random.randn(mdp.f_matrix.shape[1])
    print('Initial reward vector: {}'.format(r_vec))

    if check_grad_flag: grad_error_list=[]

    for i in range(epochs):
        dL_dr_vec = compute_grad_new(r_vec)
        if check_grad_flag:
            #grad_error_list.append(np.sqrt(sum((dL_dr_vec - approx_fprime(r_vec, compute_log_likelihood, 1e-5))**2)))
            #grad_error_list.append(check_grad(compute_log_likelihood, compute_grad, r_vec))
            grad_error_list.append(check_grad(compute_log_likelihood, get_grad, r_vec))

        # Gradient ascent
        r_vec = r_vec + learning_rate * dL_dr_vec

        if i%1==0:
            with printoptions(precision=4, suppress=True):
                print('Epoch {}; Reward vector: {}'.format(i, r_vec))
                if check_grad_flag: print('grad error: {}'.format(grad_error_list[-1]))

        if np.linalg.norm(dL_dr_vec) < threshold:
            if check_grad_flag:
                print()
                print('Max grad error: {}'.format(np.amax(np.asarray(grad_error_list))))
                print('Median grad error: {}'.format(np.median(np.asarray(grad_error_list))))
            break

    return r_vec
