import numpy as np
import warnings

from scipy.stats import expon, norm, uniform

from envs.vases_grid import VasesGrid, VasesEnvState #, print_state, str_to_state, state_to_str
from envs.utils import unique_perm, zeros_with_ones, printoptions
from envs.vases_spec import VasesEnvState2x3V2D3, VasesEnvSpec2x3V2D3, VasesEnvState2x3Broken, VasesEnvSpec2x3Broken

from value_iter_and_policy import vi_boltzmann_deterministic, vi_rational_deterministic
from occupancy_measure import compute_d_last_step_deterministic

def log_last_step_om(s_current, env, policy, p_0, horizon):
    d_last_step  = compute_d_last_step_deterministic(env, policy, p_0, horizon)
    return np.log(d_last_step[np.where(s_current)])


def policy_walk_last_state_prob(env, s_current, p_0, h, temp, n_samples,
                    step_size, r_prior, adaptive_step_size=True, verbose=True):
    '''
    Algorithm similar to BIRL that uses the last-step OM of a Boltzmann rational
    policy instead of the BIRL likelihood. Samples the reward from the posterior
    p(r | s_T, r_spec) \propto  p(s_T | \theta) * p(r | r_spec).
    '''
    i=0
    times_accepted=0
    a_list = []

    log_p = np.log(.5)
    samples = []

    r = r_prior.rvs()
    V, Q, pi = vi_boltzmann_deterministic(env, 1, env.f_matrix @ r, h, temp)

    while True:
        r_prime = np.random.normal(r, step_size)
        V, Q, pi = vi_boltzmann_deterministic(env, 1, env.f_matrix @ r_prime, h, temp)

        log_p_1 = log_last_step_om(s_current, env, pi, p_0, h) + np.sum(r_prior.logpdf(r_prime))

        # acceptance prob
        a = np.exp(log_p_1-log_p)
        # accept or reject the new sample
        if np.random.uniform()<np.amin(np.array([1, a])):
            times_accepted += 1
            samples.append(r_prime)
            r = np.copy(r_prime)
            #V = np.copy(V_prime)
            log_p = np.copy(log_p_1)
        else:
            # reject the generated sample; the new sample equals previous sample
            samples.append(r)

        # adjust step size based on acceptance prob
        # this is a bit wacky, but does make the choice of step_size less important
        # (turned off for now)
        if adaptive_step_size:
            a_list.append(a[0])
            a_running_mean = np.convolve(np.array(a_list), np.ones((25,))/25, mode='valid')
            if a_running_mean[-1]<10e-1:
                step_size = .98*step_size
            if a_running_mean[-1]>.5:
                step_size = 1.02*step_size


        if len(samples)%200==0 and verbose:
            if i!=len(samples):
                i = len(samples)
                print('samples generated: ', len(samples))
                #print(a_running_mean[-1])
                # monitoring acceptance prob; we don't want this to be very high or very low
                print(a[0])
                print(step_size)

        if len(samples)==n_samples:
            print('fraction accepted: ', times_accepted/n_samples)
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
