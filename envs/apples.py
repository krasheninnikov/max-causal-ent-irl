import numpy as np
from copy import copy, deepcopy
from itertools import product

from envs.env import Env
from envs.utils import unique_perm, Direction


class ApplesState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, agent_pos, tree_states, bucket_states, carrying_apple):
        """
        agent_pos: (orientation, x, y) tuple for the agent's location
        tree_states: Dictionary mapping (x, y) tuples to booleans.
        bucket_states: Dictionary mapping (x, y) tuples to integers.
        carrying_apple: Boolean, True if carrying an apple, False otherwise.
        """
        self.agent_pos = agent_pos
        self.tree_states = tree_states
        self.bucket_states = bucket_states
        self.carrying_apple = carrying_apple

    def __eq__(self, other):
        return isinstance(other, ApplesState) and \
            self.agent_pos == other.agent_pos and \
            self.tree_states == other.tree_states and \
            self.bucket_states == other.bucket_states

    def __hash__(self):
        def get_vals(dictionary):
            return tuple([dictionary[loc] for loc in sorted(dictionary.keys())])
        return hash(self.agent_pos + get_vals(self.tree_states) + get_vals(self.bucket_states))


class ApplesEnv(Env):
    def __init__(self, spec, compute_transitions=True):
        """
        height: Integer, height of the grid. Y coordinates are in [0, height).
        width: Integer, width of the grid. X coordinates are in [0, width).
        init_state: ApplesState, initial state of the environment
        vase_locations: List of (x, y) tuples, locations of vases
        num_vases: Integer, number of vases
        carpet_locations: Set of (x, y) tuples, locations of carpets
        feature_locations: List of (x, y) tuples, locations of features
        s: ApplesState, Current state
        nA: Integer, number of actions
        """
        self.height = spec.height
        self.width = spec.width
        self.apple_regen_probability = spec.apple_regen_probability
        self.bucket_capacity = spec.bucket_capacity
        self.init_state = deepcopy(spec.init_state)

        self.tree_locations = list(self.init_state.tree_states.keys())
        self.bucket_locations = list(self.init_state.bucket_states.keys())
        used_locations = set(self.tree_locations + self.bucket_locations)
        self.possible_agent_locations = list(filter(
            lambda pos: pos not in used_locations,
            product(range(self.width), range(self.height))))

        self.num_trees = len(self.tree_locations)
        self.num_buckets = len(self.bucket_locations)

        self.nA = 6
        self.num_features = len(self.s_to_f(self.init_state))

        self.reset()

        if compute_transitions:
            states = self.enumerate_states()
            self.make_transition_matrices(
                states, range(self.nA), self.nS, self.nA)
            self.make_f_matrix(self.nS, self.num_features)


    def enumerate_states(self):
        all_agent_positions = filter(
            lambda pos: (pos[1], pos[2]) in self.possible_agent_locations,
            product(range(4), range(self.width), range(self.height)))
        all_tree_states = map(
            lambda tree_vals: dict(zip(self.tree_locations, tree_vals)),
            product([True, False], repeat=self.num_trees))
        all_bucket_states = map(
            lambda bucket_vals: dict(zip(self.bucket_locations, bucket_vals)),
            product(range(self.bucket_capacity + 1), repeat=self.num_buckets))
        all_states = map(
            lambda x: ApplesState(*x),
            product(all_agent_positions, all_tree_states, all_bucket_states, [True, False]))

        state_num = {}
        for state in all_states:
            if state not in state_num:
                state_num[state] = len(state_num)

        self.state_num = state_num
        self.num_state = {v: k for k, v in self.state_num.items()}
        self.nS = len(state_num)

        return state_num.keys()

    def get_num_from_state(self, state):
        return self.state_num[state]

    def get_state_from_num(self, num):
        return self.num_state[num]


    def s_to_f(self, s):
        '''
        Returns features of the state:
        - Number of apples in buckets
        - Number of apples on trees
        - Whether the agent is carrying an apple
        - For each other location, whether the agent is on that location
        '''
        num_bucket_apples = sum(s.bucket_states.values())
        num_tree_apples = sum(map(int, s.tree_states.values()))
        carrying_apple = int(s.carrying_apple)
        agent_pos = s.agent_pos[1], s.agent_pos[2]  # Drop orientation
        features = [int(agent_pos == pos) for pos in self.possible_agent_locations]
        features = [num_bucket_apples, num_tree_apples, carrying_apple] + features
        return np.array(features)


    def get_next_states(self, state, action):
        '''returns the next state given a state and an action'''
        action = int(action)
        orientation, x, y = state.agent_pos
        new_orientation, new_x, new_y = state.agent_pos
        new_tree_states = deepcopy(state.tree_states)
        new_bucket_states = deepcopy(state.bucket_states)
        new_carrying_apple = state.carrying_apple

        if action == Direction.get_number_from_direction(Direction.STAY):
            pass
        elif action < len(Direction.ALL_DIRECTIONS):
            new_orientation = action
            move_x, move_y = Direction.move_in_direction_number((x, y), action)
            # New position is legal
            if (0 <= move_x < self.width and \
                0 <= move_y < self.height and \
                (move_x, move_y) in self.possible_agent_locations):
                new_x, new_y = move_x, move_y
            else:
                # Move only changes orientation, which we already handled
                pass
        elif action == 5:
            obj_pos = Direction.move_in_direction_number((x, y), orientation)
            if state.carrying_apple:
                # We always drop the apple
                new_carrying_apple = False
                # If we're facing a bucket, it goes there
                if obj_pos in new_bucket_states:
                    prev_apples = new_bucket_states[obj_pos]
                    new_bucket_states[obj_pos] = min(prev_apples + 1, 10)
            elif obj_pos in new_tree_states and new_tree_states[obj_pos]:
                new_carrying_apple = True
                new_tree_states[obj_pos] = False
            else:
                # Interact while holding nothing and not facing a tree.
                pass
        else:
            raise ValueError('Invalid action {}'.format(action))

        new_pos = new_orientation, new_x, new_y

        def make_state(prob_apples_tuple):
            prob, tree_apples = prob_apples_tuple
            trees = dict(zip(self.tree_locations, tree_apples))
            s = ApplesState(new_pos, trees, new_bucket_states, new_carrying_apple)
            return (prob, s, 0)

        # For apple regeneration, don't regenerate apples that were just picked,
        # so use the apple booleans from the original state
        old_tree_apples = [state.tree_states[loc] for loc in self.tree_locations]
        new_tree_apples = [new_tree_states[loc] for loc in self.tree_locations]
        return list(map(make_state, self.regen_apples(old_tree_apples, new_tree_apples)))

    def regen_apples(self, old_tree_apples, new_tree_apples):
        if len(old_tree_apples) == 0:
            yield (1, [])
            return
        for prob, apples in self.regen_apples(old_tree_apples[1:], new_tree_apples[1:]):
            if old_tree_apples[0]:
                yield prob, [new_tree_apples[0]] + apples
            else:
                yield prob * self.apple_regen_probability, [True] + apples
                yield prob * (1 - self.apple_regen_probability), [False] + apples


    def print_state(self, state):
        '''Renders the state.'''
        h, w = self.height, self.width
        canvas = np.zeros(tuple([2*h-1, 2*w+1]), dtype='int8')

        # cell borders
        for y in range(1, canvas.shape[0], 2):
            canvas[y, :] = 1
        for x in range(0, canvas.shape[1], 2):
            canvas[:, x] = 2

        # trees
        for (x, y), has_apple in state.tree_states.items():
            canvas[2*y, 2*x+1] = 3 if has_apple else 4

        for x, y in self.bucket_locations:
            canvas[2*y, 2*x+1] = 5

        # agent
        orientation, x, y = state.agent_pos
        canvas[2*y, 2*x+1] = 6

        black_color = '\x1b[0m'
        purple_background_color = '\x1b[0;35;85m'

        for line in canvas:
            for char_num in line:
                if char_num==0:
                    print('\u2003', end='')
                elif char_num==1:
                    print('─', end='')
                elif char_num==2:
                    print('│', end='')
                elif char_num==3:
                    print('\x1b[0;32;85m█'+black_color , end='')
                elif char_num==4:
                    print('\033[91m█'+black_color, end='')
                elif char_num==5:
                    print('\033[93m█'+black_color, end='')
                elif char_num==6:
                    orientation_char = self.get_orientation_char(orientation)
                    agent_color = '\x1b[1;42;42m' if state.carrying_apple else '\x1b[0m'
                    print(agent_color+orientation_char+black_color, end='')
            print('')

    def get_orientation_char(self, orientation):
        DIRECTION_TO_CHAR = {
            Direction.NORTH: '↑',
            Direction.SOUTH: '↓',
            Direction.WEST: '←',
            Direction.EAST: '→',
            Direction.STAY: '*'
        }
        direction = Direction.get_direction_from_number(orientation)
        return DIRECTION_TO_CHAR[direction]
