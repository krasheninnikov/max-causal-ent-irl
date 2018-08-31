import numpy as np
import sys

from envs.vases_grid import VasesGrid, VasesEnvState, print_state, str_to_state, state_to_str
from envs.vases_spec import VasesEnvState2x3V2D3, VasesEnvSpec2x3V2D3, VasesEnvState2x3Broken, VasesEnvSpec2x3Broken

from envs.irreversible_side_effects import BoxesEnv
from envs.side_effects_spec import BoxesEnvSpec6x6, BoxesEnvState6x6

from envs.utils import unique_perm, zeros_with_ones, printoptions

from last_step_om_features import om_method, compute_d_last_step_discrete

from value_iter_and_policy import vi_boltzmann, vi_boltzmann_deterministic

def forward_rl(env, r, h=40, temp=.1, steps_printed=15, current_s=None):
    '''Given an env and R, runs soft VI for h steps and rolls out the resulting policy'''
    V, Q, policy = vi_boltzmann_deterministic(env, 1, env.f_matrix @ r, h, temp) 
    
    if current_s is None: 
        env.reset()
    else:
        env.s = str_to_state(env.num_state[np.where(current_s)[0][0]])
    print_state(env.s); print()
    for i in range(steps_printed):
        a = np.random.choice(5,p=policy[env.state_num[state_to_str(env.s)],:])
        env.state_step(a)
        print_state(env.s)
        
        obs = env.s_to_f(env.s)
        
        print(obs, obs.T @ env.r_vec)
        print()

def experiment_wrapper(environment,
                     algorithm,
                     rl_algorithm,
                     horizon=22, #number of steps we assume the expert was acting previously
                     temp=1,
                     learning_rate=.1,
                     epochs = 200,
                     s_current=None,
                     uniform=False):

    if environment == "vases":
      env = VasesGrid(VasesEnvSpec2x3V2D3(), VasesEnvState2x3V2D3())
    elif environment == "boxes":
      env = BoxesEnv(BoxesEnvSpec6x6(), BoxesEnvState6x6())

    print('Initial state:')
    print_state(env.init_state)

    if not uniform:
        p_0=np.zeros(env.nS)
        p_0[env.state_num[state_to_str(env.init_state)]] = 1
    else:
        p_0=np.ones(env.nS) / env.nS
    
    if s_current is None: s_current = np.copy(p_0)
    
    if algorithm == "om":
        r_vec = om_method(env, s_current, p_0, horizon, temp, epochs, learning_rate)
        with printoptions(precision=4, suppress=True):
            print(); print('Final reward vector: ', r_vec)
        return r_vec

    if rl_algorithm == "vi":
        forward_rl(env, r_rl + r_vec)
    elif rl_algorithm == "test": 
        np.random.seed(0)
        env.reset()
        for i in range(10):
            env.print_state(env.s, env.spec)
            env.step(np.random.randint(0, env.nA))

#print(experiment_wrapper("vases", "om", "rl"))
print(experiment_wrapper("boxes", "", "test"))