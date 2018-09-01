import argparse
import csv
import datetime
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

def experiment_wrapper(env='vases',
                       algorithm='om',
                       rl_algorithm='vi',
                       horizon=22, #number of steps we assume the expert was acting previously
                       temp=1,
                       learning_rate=.1,
                       epochs=200,
                       s_current='',
                       uniform=False,
                       measures=['result']):
    if env == "vases":
        env = VasesGrid(VasesEnvSpec2x3V2D3(), VasesEnvState2x3V2D3())
    elif env == "boxes":
        env = BoxesEnv(BoxesEnvSpec6x6(), BoxesEnvState6x6())
    else:
        raise ValueError('Unknown environment: {}'.format(args.env))

    print('Initial state:')
    print_state(env.init_state)

    if not uniform:
        p_0=np.zeros(env.nS)
        p_0[env.state_num[state_to_str(env.init_state)]] = 1
    else:
        p_0=np.ones(env.nS) / env.nS
    
    if s_current == '': s_current = np.copy(p_0)
    
    if algorithm == "om":
        r_vec = om_method(env, s_current, p_0, horizon, temp, epochs, learning_rate)
        with printoptions(precision=4, suppress=True):
            print(); print('Final reward vector: ', r_vec)
        return r_vec
    else:
        raise ValueError('Unknown algorithm: {}'.format(algorithm))

    if rl_algorithm == "vi":
        forward_rl(env, r_rl + r_vec)
    elif rl_algorithm == "test": 
        np.random.seed(0)
        env.reset()
        for i in range(10):
            env.print_state(env.s, env.spec)
            env.step(np.random.randint(0, env.nA))



# Writing output for experiments

def get_filename(args):
    filename = '{}-env={}-algorithm={}-rl_algorithm={}-horizon={}-temperature={}-learning_rate={}-epochs={}-state={}-uniform_prior={}-dependent_vars={}.csv'
    filename = filename.format(
        str(datetime.datetime.now()), args.env, args.algorithm,
        args.rl_algorithm, args.horizon, args.temperature, args.learning_rate,
        args.epochs, args.state, args.uniform_prior, args.dependent_vars)
    return args.output_folder + '/' + filename

def write_output(results, indep_var, indep_vals, dependent_vars, args):
    with open(get_filename(args), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[indep_var] + dependent_vars)
        writer.writeheader()
        for indep_val, result in zip(indep_vals, results):
            row = dict(zip(dependent_vars, result))
            row[indep_var] = indep_val
            writer.writerow(row)


# Command-line arguments

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='vases',
                        help='Environment to run: one of [vases, boxes]')
    parser.add_argument('-a', '--algorithm', type=str, default='om',
                        help='Frame condition inference algorithm. Only om for now.')
    parser.add_argument('-r', '--rl_algorithm', type=str, default='vi',
                        help='Algorithm to run on the inferred reward')
    parser.add_argument('-H', '--horizon', type=str, default='22',
                        help='Number of timesteps we assume the human has been acting')
    parser.add_argument('-t', '--temperature', type=str, default='1.0',
                        help='Boltzmann rationality constant for the human')
    parser.add_argument('-l', '--learning_rate', type=str, default='0.1',
                        help='Learning rate for gradient descent')
    parser.add_argument('-p', '--epochs', type=str, default='200',
                        help='Number of epochs to run gradient descent for.')
    parser.add_argument('-s', '--state', type=str, default='',
                        help='Specifies the observed state.')
    parser.add_argument('-u', '--uniform_prior', type=str, default='false',
                        help='Whether to use a uniform prior over initial states, or to know the initial state. Either true or false.')
    parser.add_argument('-d', '--dependent_vars', type=str, default='result',
                        help='Dependent variables to measure and report')

    parser.add_argument('-o', '--output_folder', type=str, default='results',
                        help='Output folder')
    return parser.parse_args(args)


def setup_experiment(args):
    indep_vars_dict, control_vars_dict = {}, {}
    def add_to_dict(var, vals):
        if len(vals) > 1:
            indep_vars_dict[var] = vals
        else:
            control_vars_dict[var] = vals[0]

    add_to_dict('env', args.env.split(','))
    add_to_dict('algorithm', args.algorithm.split(','))
    add_to_dict('rl_algorithm', args.rl_algorithm.split(','))
    add_to_dict('horizon', [int(h) for h in args.horizon.split(',')])
    add_to_dict('temp', [float(t) for t in args.temperature.split(',')])
    add_to_dict('learning_rate', [float(lr) for lr in args.learning_rate.split(',')])
    add_to_dict('epochs', [int(epochs) for epochs in args.epochs.split(',')])
    add_to_dict('s_current', args.state.split(','))
    add_to_dict('uniform', [bool(u) for u in args.uniform_prior.split(',')])
    return indep_vars_dict, control_vars_dict, args.dependent_vars.split(',')


def main():
    args = parse_args()
    indep_vars_dict, control_vars_dict, dependent_vars = setup_experiment(args)
    # For now, restrict to zero or one independent variables, but it
    # could be generalized to two variables
    if len(indep_vars_dict) == 0:
        results = experiment_wrapper(**control_vars_dict)
        print(results)
    elif len(indep_vars_dict) == 1:
        indep_var = next(iter(indep_vars_dict.keys()))
        indep_vals = indep_vars_dict[indep_var]
        results = []
        for indep_val in indep_vals:
            experiment_args = control_vars_dict.copy()
            experiment_args[indep_var] = indep_val
            experiment_args['measures'] = dependent_vars
            results.append(experiment_wrapper(**experiment_args))
        write_output(results, indep_var, indep_vals, dependent_vars, args)
    else:
        raise ValueError('Can only support one independent variable (that is, a flag with multiple comma-separated values)')


if __name__ == '__main__':
    main()
