import argparse
import csv
import datetime
import numpy as np
import sys

from envs.vases_grid import VasesGrid, VasesEnvState
from envs.vases_spec import VASES_PROBLEMS

from envs.irreversible_side_effects import BoxesEnv
from envs.side_effects_spec import BOXES_PROBLEMS

from envs.room import RoomEnv, RoomState
from envs.room_spec import ROOM_PROBLEMS

from envs.utils import unique_perm, zeros_with_ones, printoptions

from principled_frame_cond_features import om_method, norm_distr, laplace_distr
from relative_reachability import relative_reachability_penalty

from value_iter_and_policy import vi_boltzmann, vi_boltzmann_deterministic

def forward_rl(env, r, h=40, temp=.1, steps_printed=15, current_s=None, penalize_deviation=False, relative_reachability=False):
    '''Given an env and R, runs soft VI for h steps and rolls out the resulting policy'''

    r_s = env.f_matrix @ r
    if penalize_deviation:
        r_s += np.sqrt(np.sum((env.f_matrix - env.s_to_f(env.s).T) ** 2, axis=1))
    if relative_reachability:
        r_r = relative_reachability_penalty(env, h, env.s)
        r_s -= r_r
    V, Q, policy = vi_boltzmann_deterministic(env, 1, r_s, h, temp) 
    
    if current_s is None: 
        env.reset()
    else:
        env.s = env.get_state_from_num(np.where(current_s)[0][0])

    print("Executing policy:")
    env.print_state(env.s, env.spec); print()
    # steps = [4, 1, 4, 1]
    for i in range(steps_printed):
        a = np.random.choice(env.nA, p=policy[env.get_num_from_state(env.s),:])
        # a = steps[i]
        env.step(a)
        env.print_state(env.s, env.spec)
        # print(env.get_num_from_state(env.s))
        # print(r_r[env.get_num_from_state(env.s)])
        
        obs = env.s_to_f(env.s)
        
        print(obs, obs.T @ env.r_vec)
        print()

PROBLEMS = {
    'room': ROOM_PROBLEMS,
    'vases': VASES_PROBLEMS,
    'boxes': BOXES_PROBLEMS
}

ENV_CLASSES = {
    'room': RoomEnv,
    'vases': VasesGrid,
    'boxes': BoxesEnv
}

def get_env_and_s_current(env_name, problem_name):
    if env_name not in ENV_CLASSES:
        raise ValueError('Environment {} is not one of {}'.format(env_name, list(ENV_CLASSES.keys())))
    if problem_name not in PROBLEMS[env_name]:
        raise ValueError('Problem spec {} is not one of {}'.format(problem_name, list(PROBLEMS[env_name].keys())))

    spec, cur_state = PROBLEMS[env_name][problem_name]
    env = ENV_CLASSES[env_name](spec)
    s_current = np.zeros(env.nS)
    s_current[env.get_num_from_state(cur_state)] = 1
    return env, s_current


def experiment_wrapper(env_name='vases',
                       problem_name='default',
                       algorithm='om',
                       rl_algorithm='vi',
                       prior='none',
                       horizon=22, #number of steps we assume the expert was acting previously
                       temp=1,
                       learning_rate=.1,
                       epochs=200,
                       uniform=False,
                       measures=['result']):
    env, s_current = get_env_and_s_current(env_name, problem_name)
    print('Initial state:')
    env.print_state(env.init_state, env.spec)
    print()
    if not uniform:
        p_0=np.zeros(env.nS)
        p_0[env.get_num_from_state(env.init_state)] = 1
    else:
        p_0=np.ones(env.nS) / env.nS
        
    r_vec = env.r_vec
    penalize_deviation = False
    if algorithm == "om":
        task_weight = 2 
        safety_weight = 1
        if prior == "gaussian":
            r_prior = norm_distr(env.r_vec, 1)
        elif prior == "laplace":
            r_prior = laplace_distr(env.r_vec, 1)
        elif prior == "none":
            r_prior = None

        om_vec = om_method(env, s_current, p_0, horizon, temp, epochs, learning_rate, r_prior)
        om_vec = om_vec / np.linalg.norm(om_vec)
        r_vec = task_weight * r_vec + safety_weight * om_vec
        with printoptions(precision=4, suppress=True):
            print(); print('Final reward vector: ', r_vec)
    elif algorithm == "pen_dev":
        penalize_deviation = True
    elif algorithm == "pass":
        pass
    else:
        raise ValueError('Unknown algorithm: {}'.format(algorithm))

    if rl_algorithm == "vi":
        forward_rl(env, r_vec, current_s=s_current, penalize_deviation=penalize_deviation)
    elif rl_algorithm == "test": 
        np.random.seed(0)
        env.reset()
        for a in [0, 1, 2, 2, 2, 1]:
            env.step(a)
            env.print_state(env.s, env.spec)
    elif rl_algorithm == "relative_reachability":
        forward_rl(env, r_vec, current_s=s_current, penalize_deviation=penalize_deviation, relative_reachability=True)

    return r_vec


# Writing output for experiments
def get_filename(args):
    filename = '{}-env={}-algorithm={}-rl_algorithm={}-state={}-prior={}-horizon={}-temperature={}-learning_rate={}-epochs={}-uniform_prior={}-dependent_vars={}.csv'
    filename = filename.format(
        str(datetime.datetime.now()), args.env, args.algorithm,
        args.rl_algorithm, args.state, args.prior, args.horizon, args.temperature, args.learning_rate, args.epochs, args.uniform_prior, args.dependent_vars)
    return args.output_folder + '/' + filename

def write_output(results, indep_var, indep_vals, dependent_vars, args):
    with open(get_filename(args), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[indep_var] + dependent_vars)
        writer.writeheader()
        for indep_val, result in zip(indep_vals, results):
            row = {}
            row[dependent_vars[0]] = result
            row[indep_var] = indep_val
            writer.writerow(row)


# Command-line arguments
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, default='room',
                        help='Environment to run: one of [vases, boxes, room]')
    parser.add_argument('-s', '--problem_spec', type=str, default='simple',
                        help='The name of the problem specification to solve.')
    parser.add_argument('-a', '--algorithm', type=str, default='pass',
                        help='Frame condition inference algorithm.')
    parser.add_argument('-r', '--rl_algorithm', type=str, default='vi',
                        help='Algorithm to run on the inferred reward')
    parser.add_argument('-i', '--prior', type=str, default='none',
                        help='Prior to use for occupancy measure')
    parser.add_argument('-H', '--horizon', type=str, default='22',
                        help='Number of timesteps we assume the human has been acting')
    parser.add_argument('-t', '--temperature', type=str, default='1.0',
                        help='Boltzmann rationality constant for the human')
    parser.add_argument('-l', '--learning_rate', type=str, default='0.1',
                        help='Learning rate for gradient descent')
    parser.add_argument('-p', '--epochs', type=str, default='200',
                        help='Number of gradient descent steps to take.')
    parser.add_argument('-u', '--uniform_prior', type=str, default='False',
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

    add_to_dict('env_name', args.env.split(','))
    add_to_dict('problem_name', args.problem_spec.split(','))
    add_to_dict('algorithm', args.algorithm.split(','))
    add_to_dict('rl_algorithm', args.rl_algorithm.split(','))
    add_to_dict('prior', args.prior.split(','))
    add_to_dict('horizon', [int(h) for h in args.horizon.split(',')])
    add_to_dict('temp', [float(t) for t in args.temperature.split(',')])
    add_to_dict('learning_rate', [float(lr) for lr in args.learning_rate.split(',')])
    add_to_dict('epochs', [int(epochs) for epochs in args.epochs.split(',')])
    add_to_dict('uniform', [u != "False" for u in args.uniform_prior.split(',')])
    return indep_vars_dict, control_vars_dict, args.dependent_vars.split(',')


def main():
    args = parse_args()
    indep_vars_dict, control_vars_dict, dependent_vars = setup_experiment(args)
    print(indep_vars_dict, control_vars_dict, dependent_vars)
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
