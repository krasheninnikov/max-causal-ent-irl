import argparse
import csv
import datetime
import numpy as np
import sys
from scipy.stats import uniform as uniform_distr

from envs.vases_grid import VasesGrid, VasesEnvState
from envs.vases_spec import VASES_PROBLEMS

from envs.irreversible_side_effects import BoxesEnv
from envs.side_effects_spec import BOXES_PROBLEMS

from envs.room import RoomEnv, RoomState
from envs.room_spec import ROOM_PROBLEMS

from envs.apples import ApplesEnv, ApplesState
from envs.apples_spec import APPLES_PROBLEMS

from envs.utils import unique_perm, zeros_with_ones, printoptions

from sampling_one_s_mceirl import policy_walk_last_state_prob
from principled_frame_cond_features import om_method, norm_distr, laplace_distr
from relative_reachability import relative_reachability_penalty

from value_iter import value_iter


def forward_rl(env, r_planning, r_true, h=40, temp=.1, last_steps_printed=3,
               current_s=None, weight=1, penalize_deviation=False,
               relative_reachability=False, print_level=1):
    '''Given an env and R, runs soft VI for h steps and rolls out the resulting policy'''

    r_s = env.f_matrix @ r_planning
    if penalize_deviation:
        r_s += weight * np.sqrt(np.sum((env.f_matrix - env.s_to_f(env.s).T) ** 2, axis=1))
    if relative_reachability:
        r_r = relative_reachability_penalty(env, h, env.s)
        r_s -= weight * r_r

    # For evaluation, plan optimally instead of Boltzmann-rationally
    V, Q, policy = value_iter(env, 1, r_s, h, temperature=None)

    if current_s is None:
        env.reset()
    else:
        env.s = env.get_state_from_num(np.where(current_s)[0][0])

    if print_level >= 1:
        print("Executing the policy from state:")
        env.print_state(env.s); print()

    # steps = [4, 1, 4, 1]
    total_reward = 0
    if print_level >= 1:
        print('Last {} of the {} rolled out steps:'.format(last_steps_printed, h))
    for i in range(h):
        a = np.random.choice(env.nA, p=policy[env.get_num_from_state(env.s),:])
        # a = steps[i]
        obs, reward, done, info = env.step(a, r_vec=r_true)
        total_reward += reward

        if print_level >= 1 and i>=(h-last_steps_printed):
            env.print_state(env.s)
            print()

    return total_reward

PROBLEMS = {
    'room': ROOM_PROBLEMS,
    'vases': VASES_PROBLEMS,
    'apples': APPLES_PROBLEMS,
    'boxes': BOXES_PROBLEMS
}

ENV_CLASSES = {
    'room': RoomEnv,
    'vases': VasesGrid,
    'apples': ApplesEnv,
    'boxes': BoxesEnv
}

def get_problem_parameters(env_name, problem_name):
    if env_name not in ENV_CLASSES:
        raise ValueError('Environment {} is not one of {}'.format(env_name, list(ENV_CLASSES.keys())))
    if problem_name not in PROBLEMS[env_name]:
        raise ValueError('Problem spec {} is not one of {}'.format(problem_name, list(PROBLEMS[env_name].keys())))

    spec, cur_state, r_task, r_true = PROBLEMS[env_name][problem_name]
    env = ENV_CLASSES[env_name](spec)
    s_current = np.zeros(env.nS)
    s_current[env.get_num_from_state(cur_state)] = 1
    return env, s_current, r_task, r_true


def get_r_prior(prior, reward_center):
    if prior == "gaussian":
        return norm_distr(reward_center, 1)
    elif prior == "laplace":
        return laplace_distr(reward_center, 1)
    elif prior == "uniform":
        return None
    else:
        raise ValueError('Unknown prior {}'.format(prior))


def experiment_wrapper(env_name='vases',
                       problem_spec='default',
                       inference_algorithm='mceirl',
                       combination_algorithm='add_rewards',
                       prior='gaussian',
                       horizon=25,
                       temperature=1,
                       learning_rate=.1,
                       inferred_weight=1,
                       epochs=200,
                       uniform_prior=False,
                       measures=['final_reward'],
                       n_samples=1000,
                       mcmc_burn_in=400,
                       step_size=.01,
                       gamma=1,
                       seed=0,
                       print_level=1):
    # Check the parameters so that we fail fast
    assert inference_algorithm in ['mceirl', 'sampling', 'deviation', 'reachability', 'pass']
    assert combination_algorithm in ['add_rewards', 'use_prior']
    assert prior in ['gaussian', 'laplace', 'uniform']
    assert all((measure in ['true_reward', 'final_reward'] for measure in measures))
    if combination_algorithm == 'use_prior':
        assert inference_algorithm in ['mceirl', 'sampling']

    np.random.seed(seed)
    env, s_current, r_task, r_true = get_problem_parameters(env_name, problem_spec)

    if print_level >= 1:
        print('Initial state:')
        env.print_state(env.init_state)
        print()

    if not uniform_prior:
        p_0=np.zeros(env.nS)
        p_0[env.get_num_from_state(env.init_state)] = 1
    else:
        p_0=np.ones(env.nS) / env.nS

    deviation = inference_algorithm == "deviation"
    reachability = inference_algorithm == "reachability"
    reward_center = r_task if combination_algorithm == "use_prior" else np.zeros(env.num_features)
    r_prior = get_r_prior(prior, reward_center)

    # TODO: Normalization of task and inferred rewards
    if inference_algorithm == "mceirl":
        r_inferred = om_method(env, s_current, p_0, horizon, temperature, epochs, learning_rate, r_prior)
        # r_inferred = r_inferred / np.linalg.norm(r_inferred)
    elif inference_algorithm == "sampling":
        r_samples = policy_walk_last_state_prob(
            env, s_current, p_0, horizon, temperature, n_samples, step_size,
            r_prior, gamma, adaptive_step_size=True)
        r_inferred = np.mean(r_samples[mcmc_burn_in::], axis=0)
    elif inference_algorithm in ["deviation", "reachability", "pass"]:
        r_inferred = None
    else:
        raise ValueError('Unknown inference algorithm: {}'.format(inference_algorithm))

    if print_level >= 1 and r_inferred is not None:
        with printoptions(precision=4, suppress=True):
            print(); print('Inferred reward vector: ', r_inferred)

    # Run forward RL to evaluate
    if combination_algorithm == "add_rewards":
        r_final = r_task
        if r_inferred is not None:
            r_final = r_task + inferred_weight * r_inferred
        true_reward_obtained = forward_rl(env, r_final, r_true, h=horizon, current_s=s_current, weight=inferred_weight, penalize_deviation=deviation, relative_reachability=reachability, print_level=print_level)
    elif combination_algorithm == "use_prior":
        assert r_inferred is not None
        assert (not deviation) and (not reachability)
        r_final = r_inferred
        true_reward_obtained = forward_rl(env, r_final, r_true, h=horizon, current_s=s_current, penalize_deviation=False, relative_reachability=False, print_level=print_level)
    else:
        raise ValueError('Unknown combination algorithm: {}'.format(combination_algorithm))

    def get_measure(measure):
        if measure == 'final_reward':
            return r_final
        elif measure == 'true_reward':
            return true_reward_obtained
        else:
            raise ValueError('Unknown measure {}'.format(measure))

    return [get_measure(measure) for measure in measures]



# The command line parameters that should be included in the filename of the
# file summarizing the results.
PARAMETERS = [
    ('-e', '--env_name', 'room', None,
     'Environment to run: one of [vases, boxes, room]'),
    ('-p', '--problem_spec', 'simple', None,
     'The name of the problem specification to solve.'),
    ('-i', '--inference_algorithm', 'pass', None,
     'Frame condition inference algorithm: one of [mceirl, sampling, deviation, reachability, pass].'),
    ('-c', '--combination_algorithm', 'add_rewards', None,
     'How to combine the task reward and inferred reward for forward RL: one of [add_rewards, use_prior]. use_prior only has an effect if algorithm is mceirl or sampling.'),
    ('-r', '--prior', 'gaussian', None,
     'Prior on the inferred reward function: one of [gaussian, laplace, uniform]. Centered at zero if combination_algorithm is add_rewards, and at the task reward if combination_algorithm is use_prior. Only has an effect if inference_algorithm is mceirl or sampling.'),
    ('-H', '--horizon', '20', int,
     'Number of timesteps we assume the human has been acting.'),
    ('-t', '--temperature', '1.0', float,
     'Boltzmann rationality constant for the human. Note this is temperature, which is the inverse of beta.'),
    ('-l', '--learning_rate', '0.1', float,
     'Learning rate for gradient descent. Applies when inference_algorithm is mceirl.'),
    ('-k', '--inferred_weight', '1', float,
     'Weight for the inferred reward when adding task and inferred rewards. Applies if combination_algorithm is add_rewards.'),
    ('-m', '--epochs', '20', int,
     'Number of gradient descent steps to take.'),
    ('-u', '--uniform_prior', 'False', lambda x: x != 'False',
     'Whether to use a uniform prior over initial states, or to know the initial state. Either true or false.'),
    ('-d', '--dependent_vars', 'final_reward', None,
     'Dependent variables to measure and report'),
    ('-n', '--n_samples', '1000', int,
     'Number of samples to generate with MCMC'),
    ('-b', '--mcmc_burn_in', '400', int,
     'Number of samples to ignore at the start'),
    ('-z', '--step_size', '0.01', float,
     'Step size for computing neighbor reward functions. Only has an effect if inference_algorithm is sampling.'),
    ('-g', '--gamma', '1.0', float,
     'Discounting rate for infinite horizon discounted algorithms.'),
    ('-s', '--seed', '0', int,
     'Random seed.')
]

# Writing output for experiments
def get_filename(args):
    # Drop the '--' in front of the names
    param_names = [name[2:] for _, name, _, _, _ in PARAMETERS]
    param_values = [args.__dict__[name] for name in param_names]

    filename = '{}-' + '={}-'.join(param_names) + '={}.csv'
    filename = filename.format(str(datetime.datetime.now()), *param_values)
    return args.output_folder + '/' + filename

def write_output(results, indep_var, indep_vals, dependent_vars, args):
    with open(get_filename(args), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[indep_var] + dependent_vars)
        writer.writeheader()
        for indep_val, result in zip(indep_vals, results):
            row = {}
            row[indep_var] = indep_val
            for dependent_var, dependent_val in zip(dependent_vars, result):
                row[dependent_var] = dependent_val
            writer.writerow(row)


# Command-line arguments
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    for name, long_name, default, _, help_str in PARAMETERS:
        parser.add_argument(name, long_name, type=str, default=default, help=help_str)

    # Parameters that shouldn't be included in the filename.
    parser.add_argument('-o', '--output_folder', type=str, default='results',
                        help='Output folder')
    return parser.parse_args(args)


def setup_experiment(args):
    indep_vars_dict, control_vars_dict = {}, {}

    for _, var, _, fn, _ in PARAMETERS:
        var = var[2:]
        if var == 'dependent_vars': continue
        if fn is None: fn = lambda x: x

        vals = [fn(x) for x in args.__dict__[var].split(',')]
        if len(vals) > 1:
            indep_vars_dict[var] = vals
        else:
            control_vars_dict[var] = vals[0]

    return indep_vars_dict, control_vars_dict, args.dependent_vars.split(',')


def main():
    if sys.platform == "win32":
        import colorama; colorama.init()

    args = parse_args()
    print(args)
    indep_vars_dict, control_vars_dict, dependent_vars = setup_experiment(args)
    # print(indep_vars_dict, control_vars_dict, dependent_vars)
    # For now, restrict to zero or one independent variables, but it
    # could be generalized to two variables
    if len(indep_vars_dict) == 0:
        results = experiment_wrapper(measures=dependent_vars, **control_vars_dict)
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
