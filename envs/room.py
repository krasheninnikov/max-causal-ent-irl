import numpy as np
from collections import defaultdict
from copy import copy, deepcopy
from gym import spaces

from envs.env import DeterministicEnv
from envs.utils import unique_perm, Direction, all_boolean_assignments


class RoomState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, agent_pos, vase_states):
        """
        agent_pos: (x, y) tuple for the agent's location
        vase_states: Dictionary mapping (x, y) tuples to booleans, where True
            means that the vase is intact
        """
        self.agent_pos = agent_pos
        self.vase_states = vase_states

    def __eq__(self, other):
        return isinstance(other, RoomState) and \
            self.agent_pos == other.agent_pos and \
            self.vase_states == other.vase_states

    def __hash__(self):
        return hash(self.agent_pos) + hash(tuple(self.vase_states.values()))

    def __str__(self):
        return '<Agent: {}, Vases: {}>'.format(self.agent_pos, self.vase_states)

    def __repr__(self):
        return str(self)


class RoomEnv(DeterministicEnv):
    def __init__(self, spec, compute_transitions=True):
        """
        height: Integer, height of the grid. Y coordinates are in [0, height).
        width: Integer, width of the grid. X coordinates are in [0, width).
        init_state: RoomState, initial state of the environment
        vase_locations: List of (x, y) tuples, locations of vases
        num_vases: Integer, number of vases
        carpet_locations: Set of (x, y) tuples, locations of carpets
        feature_locations: List of (x, y) tuples, locations of features
        s: RoomState, Current state
        nA: Integer, number of actions
        """
        self.height = spec.height
        self.width = spec.width
        self.init_state = deepcopy(spec.init_state)
        self.vase_locations = list(self.init_state.vase_states.keys())
        self.num_vases = len(self.vase_locations)
        self.carpet_locations = set(spec.carpet_locations)
        self.feature_locations = list(spec.feature_locations)
        self.s = deepcopy(self.init_state)
        self.spec = None  # TODO: Remove this line? test.py might use it?

        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)

        self.num_features = len(self.s_to_f(self.init_state))
        # TODO: Need to figure out what this is doing
        self.r_vec = np.array([0,1,0,1,0], dtype='float32')
        self.observation_space = spaces.Box(low=0, high=255, shape=self.r_vec.shape, dtype=np.float32)

        self.timestep = 0

        if compute_transitions:
            states = self.enumerate_states()
            self.make_transition_matrices(
                states, range(self.nA), self.nS, self.nA)
            self.make_f_matrix(self.nS, self.num_features)


    def enumerate_states(self):
        state_num = {}

        # Possible vase states
        for vase_intact_bools in all_boolean_assignments(self.num_vases):
            vase_states = dict(zip(self.vase_locations, vase_intact_bools))
            # Possible agent positions
            for y in range(self.height):
                for x in range(self.width):
                    pos = (x, y)
                    if pos in vase_states and vase_states[pos]:
                        # Can't have the agent on an intact vase
                        continue
                    state = RoomState(pos, vase_states)
                    if state not in state_num:
                        state_num[state] = len(state_num)

        self.state_num = state_num
        self.num_state = {v: k for k, v in self.state_num.items()}
        self.nS = len(state_num)

        return state_num.keys()

    def get_num_from_state(self, state):
        return self.state_num[state]

    def get_state_from_num(self, state_num_id):
        return self.num_state[state_num_id]


    def s_to_f(self, s):
        '''
        Returns features of the state:
        - Number of broken vases
        - Whether the agent is on a carpet
        - For each feature location, whether the agent is on that location
        '''
        num_broken_vases = list(s.vase_states.values()).count(False)
        carpet_feature = int(s.agent_pos in self.carpet_locations)
        features = [int(s.agent_pos == fpos) for fpos in self.feature_locations]
        features = np.array([num_broken_vases, carpet_feature] + features)
        return features


    def reset(self):
        self.timestep = 0
        self.s = deepcopy(self.init_state)

        obs = self.s_to_f(self.s)
        return np.array(obs, dtype='float32').flatten() #, obs.T @ self.r_vec, False, defaultdict(lambda : '')


    def state_step(self, action, state=None):
        '''returns the next state given a state and an action'''
        action = int(action)
        if state==None: state = self.s
        new_x, new_y = Direction.move_in_direction_number(state.agent_pos, action)
        # New position is still in bounds:
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            new_x, new_y = state.agent_pos
        new_agent_pos = new_x, new_y
        new_vase_states = deepcopy(state.vase_states)
        if new_agent_pos in new_vase_states:
            new_vase_states[new_agent_pos] = False  # Break the vase
        return RoomState(new_agent_pos, new_vase_states)


    def step(self, action):
        '''
        given an action, takes a step from self.s, updates self.s and returns:
        - the observation (features of the next state)
        - the associated reward
        - done, the indicator of completed episode
        - info
        '''
        self.s = self.state_step(action)
        self.timestep+=1

        obs = self.s_to_f(self.s)
        done = False
        if self.timestep>500: done=True

        info = defaultdict(lambda : '')
        return np.array(obs, dtype='float32'), np.array(obs.T @ self.r_vec), np.array(done, dtype='bool'), info


    def close(self):
        self.reset()


    def seed(self, seed=None):
        pass


    def print_state(self, state, spec=None):
        '''
        Renders the state. Each tile in the gridworld corresponds to a 2x2 cell in
        the rendered state.
        - Green tiles correspond to vases
        - Red tiles correspond to broken vases
        - Brown tiles correspond to tables
        - Purple tiles correspond to tablecloths
        - The arrow and its direction correspond to the agent and its rotation. The
        background color of the arrow corresponds to the object the agent is
        carrying. The agent is rendered in the same subcell as tables are since
        the agent and the table are never in the same cell.
        '''
        h, w = self.height, self.width
        canvas = np.zeros(tuple([2*h-1, 3*w+1]), dtype='int8')

        # cell borders
        for y in range(1, canvas.shape[0], 2):
            canvas[y, :] = 1
        for x in range(0, canvas.shape[1], 3):
            canvas[:, x] = 2

        # vases
        for x, y in self.vase_locations:
            if state.vase_states[(x, y)]:
                canvas[2*y, 3*x+1] = 4
            else:
                canvas[2*y, 3*x+1] = 6

        # agent
        x, y = state.agent_pos
        canvas[2*y, 3*x + 2] = 3

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
                    print('\x1b[0;33;85m█'+black_color, end='')
                elif char_num==4:
                    print('\x1b[0;32;85m█'+black_color , end='')
                elif char_num==5:
                    print(purple_background_color+'█'+black_color, end='')
                elif char_num==6:
                    print('\033[91m█'+black_color, end='')
            print('')
