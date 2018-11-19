import numpy as np
from copy import copy, deepcopy
from itertools import product

from envs.env import DeterministicEnv
from envs.utils import unique_perm, Direction


class BatteriesState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, agent_pos, train_pos, train_life, battery_present, carrying_battery):
        """
        agent_pos: (x, y) tuple for the agent's location
        vase_states: Dictionary mapping (x, y) tuples to booleans, where True
            means that the vase is intact
        """
        self.agent_pos = agent_pos
        self.train_pos = train_pos
        self.train_life = train_life
        self.battery_present = battery_present
        self.carrying_battery = carrying_battery

    def is_valid(self):
        pos = self.agent_pos
        # Can't be standing on a battery and not carrying a battery
        if pos in self.battery_present and self.battery_present[pos] and not self.carrying_battery:
            return False
        return True

    def __eq__(self, other):
        return isinstance(other, BatteriesState) and \
            self.agent_pos == other.agent_pos and \
            self.train_pos == other.train_pos and \
            self.train_life == other.train_life and \
            self.battery_present == other.battery_present and \
            self.carrying_battery == other.carrying_battery

    def __hash__(self):
        def get_vals(dictionary):
            return tuple([dictionary[loc] for loc in sorted(dictionary.keys())])
        return hash(self.agent_pos + self.train_pos + (self.train_life,) + get_vals(self.battery_present) + (self.carrying_battery,))

    def __str__(self):
        return '{}  {}  {}  {}  {}'.format(self.agent_pos, self.train_pos, self.train_life, self.battery_present, self.carrying_battery)

    def __repr__(self):
        return str(self)


class BatteriesEnv(DeterministicEnv):
    def __init__(self, spec, compute_transitions=True):
        """
        height: Integer, height of the grid. Y coordinates are in [0, height).
        width: Integer, width of the grid. X coordinates are in [0, width).
        init_state: BatteriesState, initial state of the environment
        vase_locations: List of (x, y) tuples, locations of vases
        num_vases: Integer, number of vases
        carpet_locations: Set of (x, y) tuples, locations of carpets
        feature_locations: List of (x, y) tuples, locations of features
        s: BatteriesState, Current state
        nA: Integer, number of actions
        """
        self.height = spec.height
        self.width = spec.width
        self.init_state = deepcopy(spec.init_state)
        self.battery_locations = sorted(list(self.init_state.battery_present.keys()))
        self.num_batteries = len(self.battery_locations)
        self.feature_locations = list(spec.feature_locations)
        self.train_transition = spec.train_transition
        self.train_locations = list(self.train_transition.keys())
        assert set(self.train_locations) == set(self.train_transition.values())

        self.default_action = Direction.get_number_from_direction(Direction.STAY)
        self.nA = 5
        self.num_features = len(self.s_to_f(self.init_state))

        self.reset()

        if compute_transitions:
            states = self.enumerate_states()
            self.make_transition_matrices(
                states, range(self.nA), self.nS, self.nA)
            self.make_f_matrix(self.nS, self.num_features)


    def enumerate_states(self):
        state_num = {}
        all_agent_positions = product(range(self.width), range(self.height))
        all_battery_states = map(
            lambda battery_vals: dict(zip(self.battery_locations, battery_vals)),
            product([True, False], repeat=self.num_batteries))
        all_states = map(
            lambda x: BatteriesState(*x),
            product(all_agent_positions, self.train_locations, range(10), all_battery_states, [True, False]))
        all_states = filter(lambda state: state.is_valid(), all_states)

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
        - Number of batteries
        - Whether the train is still alive
        - For each train location, whether the train is at that location
        - For each feature location, whether the agent is on that location
        '''
        num_batteries = list(s.battery_present.values()).count(True)
        train_dead_feature = int(s.train_life == 0)
        train_pos_features = [int(s.train_pos == pos) for pos in self.train_locations]
        loc_features = [int(s.agent_pos == fpos) for fpos in self.feature_locations]
        features = train_pos_features + loc_features
        features = [num_batteries, train_dead_feature] + features
        return np.array(features)


    def get_next_state(self, state, action):
        '''returns the next state given a state and an action'''
        action = int(action)
        new_x, new_y = Direction.move_in_direction_number(state.agent_pos, action)
        # New position is still in bounds:
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            new_x, new_y = state.agent_pos
        new_agent_pos = new_x, new_y

        new_train_pos, new_train_life = state.train_pos, state.train_life
        new_battery_present = deepcopy(state.battery_present)
        new_carrying_battery = state.carrying_battery
        if new_agent_pos == state.train_pos and state.carrying_battery:
            new_train_life = 10
            new_carrying_battery = False

        if new_train_life > 0:
            new_train_pos = self.train_transition[state.train_pos]
            new_train_life -= 1

        if new_agent_pos in state.battery_present and state.battery_present[new_agent_pos] and not state.carrying_battery:
            new_carrying_battery = True
            new_battery_present[new_agent_pos] = False

        result = BatteriesState(new_agent_pos, new_train_pos, new_train_life, new_battery_present, new_carrying_battery)
        if result not in self.state_num:
            print('{} and {}'.format(state, action))
        return result


    def print_state(self, state):
        '''Renders the state.'''
        h, w = self.height, self.width
        grid = [[' '] * w for _ in range(h)]
        x, y = state.agent_pos
        grid[y][x] = 'A'
        x, y = state.train_pos
        grid[y][x] = 'T'
        for (x, y), val in state.battery_present.items():
            if val:
                grid[y][x] = 'B'
        print('\n'.join(['|'.join(row) for row in grid]))

        print('carrying_battery: ', state.carrying_battery)

        # canvas = np.zeros(tuple([2*h-1, 3*w+1]), dtype='int8')

        # # cell borders
        # for y in range(1, canvas.shape[0], 2):
        #     canvas[y, :] = 1
        # for x in range(0, canvas.shape[1], 3):
        #     canvas[:, x] = 2

        # # vases
        # for x, y in self.vase_locations:
        #     if state.vase_states[(x, y)]:
        #         canvas[2*y, 3*x+1] = 4
        #     else:
        #         canvas[2*y, 3*x+1] = 6

        # # agent
        # x, y = state.agent_pos
        # canvas[2*y, 3*x + 2] = 3

        # black_color = '\x1b[0m'
        # purple_background_color = '\x1b[0;35;85m'

        # for line in canvas:
        #     for char_num in line:
        #         if char_num==0:
        #             print('\u2003', end='')
        #         elif char_num==1:
        #             print('─', end='')
        #         elif char_num==2:
        #             print('│', end='')
        #         elif char_num==3:
        #             print('\x1b[0;33;85m█'+black_color, end='')
        #         elif char_num==4:
        #             print('\x1b[0;32;85m█'+black_color , end='')
        #         elif char_num==5:
        #             print(purple_background_color+'█'+black_color, end='')
        #         elif char_num==6:
        #             print('\033[91m█'+black_color, end='')
        #     print('')
