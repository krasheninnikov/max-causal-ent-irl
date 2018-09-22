import numpy as np
import warnings

from math import exp
from scipy.stats import expon

from envs.vases_grid import VasesGrid, VasesEnvState #, print_state, str_to_state, state_to_str
from envs.utils import unique_perm, zeros_with_ones, printoptions
from envs.vases_spec import VasesEnvState2x3V2D3, VasesEnvSpec2x3V2D3, VasesEnvState2x3Broken, VasesEnvSpec2x3Broken

from value_iter import value_iter
from principled_frame_cond_features import compute_d_last_step


def policy_walk_last_state_prob(
        env, s_current, p_0, h, temp, n_samples, step_size, r_prior, gamma=1,
        adaptive_step_size=False, print_level=1):
    '''
    Algorithm similar to BIRL that uses the last-step OM of a Boltzmann rational
    policy instead of the BIRL likelihood. Samples the reward from the posterior
    p(r | s_T, r_spec) \propto  p(s_T | \theta) * p(r | r_spec).
    '''

    def log_last_step_om(policy):
        d_last_step  = compute_d_last_step(env, policy, p_0, h)
        return np.log(d_last_step[s_current])

    def log_probability(r_vec, verbose=False):
        V, Q, pi = value_iter(env, gamma, env.f_matrix @ r_vec, h, temp)
        log_p = log_last_step_om(pi)

        log_prior = 0
        if r_prior is not None:
            log_prior = np.sum(r_prior.logpdf(r_vec))

        if verbose:
            print('Log prior: {}\nLog prob:  {}\nTotal:     {}'.format(
                log_prior, log_p, log_p + log_prior))
        return log_p + log_prior

    times_accepted = 0
    a_list = []
    samples = []

    if r_prior is None:
        r = .01*np.random.randn(env.num_features)
    else:
        r = r_prior.rvs()

    # probability of the initial reward
    log_p = log_probability(r, verbose=(print_level >= 1))

    while len(samples) < n_samples:
        verbose = (print_level >= 1) and (len(samples) % 200 == 199)
        if verbose:
            print('\nGenerating sample {}'.format(len(samples) + 1))

        r_prime = np.random.normal(r, step_size)
        log_p_1 = log_probability(r_prime, verbose=verbose)

        # Accept or reject the new sample
        # If we reject, the new sample is the previous sample
        acceptance_probability = exp(log_p_1-log_p)
        if np.random.uniform() < acceptance_probability:
            times_accepted += 1
            r, log_p = r_prime, log_p_1
        samples.append(r)

        # adjust step size based on acceptance prob
        # this is a bit wacky, but does make the choice of step_size less important
        if adaptive_step_size:
            a_list.append(acceptance_probability)
            a_running_mean = np.convolve(np.array(a_list), np.ones((5,))/5, mode='valid')
            if a_running_mean[-1]<10e-1 and step_size>10e-6:
                step_size = .98*step_size
            if a_running_mean[-1]>.5:
                step_size = 1.02*step_size


        if verbose:
            # Acceptance probability should not be very high or very low
            print(acceptance_probability)
            # print(a_running_mean[-1])
            # print(step_size)

    if print_level >= 1:
        print('Done! Accepted {} of samples'.format(times_accepted/n_samples))
    return samples


class neg_exp_distr(object):
    '''
    a wrapper to not get confused with the negative exp distribuion,
    as scipy doesn't have a good way to do *negative* exp
    '''
    def __init__(self, mode, scale=1):
        self.mode = mode
        self.scale = scale
        self.distribution = expon(loc=-mode, scale=scale)

    def rvs(self):
        '''sample'''
        return -self.distribution.rvs()

    def pdf(self, x):
        return self.distribution.pdf(-x)

    def logpdf(self, x):
        return self.distribution.logpdf(-x)
