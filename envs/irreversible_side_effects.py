import numpy as np
from collections import defaultdict
from copy import copy, deepcopy
from envs.env import DeterministicEnv
from envs.utils import unique_perm, zeros_with_ones

shift_coord_array = [(0, -1), (1, 0), (0, 1), (-1, 0)]
# [west, south, east, north]

class BoxesEnvState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, a_pos, b_pos):
        self.a_pos = np.array(a_pos, dtype='bool')
        self.b_pos = np.array(b_pos, dtype='bool')

        self.a_pos_tuple = tuple(self.a_pos.flatten())
        self.b_pos_tuple = tuple(self.b_pos.flatten())

    def __eq__(self, other):
        return isinstance(other, BoxesEnvState) and \
            self.a_pos_tuple == other.a_pos_tuple and \
            self.b_pos_tuple == other.b_pos_tuple

    def __hash__(self):
        return hash(self.a_pos_tuple + self.b_pos_tuple)


class BoxesEnv(DeterministicEnv):
    def __init__(self, spec, f_include_masks=False, compute_transitions=True):
        self.spec = spec
        self.init_state = deepcopy(spec.init_state)

        self.default_action = 4  # Direction.STAY
        self.nA = 5

        self.nF = 4
        self.f_include_masks = f_include_masks
        f_len = len(self.s_to_f(self.init_state))
        self.num_features = f_len
        self.reset()

        if compute_transitions:
            states = self.enumerate_states()
            self.make_transition_matrices(
                states, range(self.nA), self.nS, self.nA)
            self.make_f_matrix(self.nS, self.num_features)

    def enumerate_states(self):
        state_num = {}
        n_a_pos = np.sum(self.spec.agent_mask)
        n_b_pos = np.sum(self.spec.box_mask)

        # Possible agent positions
        for a_pos in unique_perm(zeros_with_ones(n_a_pos, 1)):
            agent_mask_pos = np.zeros_like(self.spec.agent_mask.flatten())
            np.put(agent_mask_pos, np.where(self.spec.agent_mask.flatten()), a_pos)
            agent_mask_pos = agent_mask_pos.reshape(self.spec.agent_mask.shape)

            # Possible box positions
            for b_pos in unique_perm(zeros_with_ones(n_b_pos, self.spec.n_b)):
                b_mask_pos = np.zeros_like(self.spec.box_mask.flatten())
                np.put(b_mask_pos, np.where(self.spec.box_mask.flatten()), b_pos)
                b_mask_pos = b_mask_pos.reshape(self.spec.box_mask.shape)

                if not b_mask_pos[np.where(agent_mask_pos)[0][0]][np.where(agent_mask_pos)[1][0]]:
                    state = BoxesEnvState(agent_mask_pos, b_mask_pos)

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


    def s_to_f(self, s, include_masks=None):
        '''
        returns features of the state:
        - Number of boxes adjacent to zero or more walls
        - Number of boxes adjacent to one or more walls
        - Number of boxes adjacent to two or more walls
        - Whether agent is on the goal state or not
        '''
        if include_masks==None:
            include_masks = self.f_include_masks
        f = np.zeros(self.nF)

        # Iterate over box positions 
        for r in range(s.b_pos.shape[0]):
            for c in range(s.b_pos.shape[1]):
                if s.b_pos[r][c]:
                    # Count adjacent walls
                    surround = 0
                    for shift in shift_coord_array:
                        if r+shift[0] < len(s.b_pos) and r+shift[0] >= 0 and c+shift[1] < len(s.b_pos[r+shift[0]]) and c+shift[1] >= 0:
                            if self.spec.wall_mask[r+shift[0]][c+shift[1]]:
                                surround += 1
                    for w in range(3):
                        f[w] += 1 if surround >= w else 0
        f[-1] = np.sum(np.logical_and(s.a_pos, self.spec.goal_mask))

        f_mask = np.array([])
        if include_masks:
            f_mask = np.array(list(self.state_to_str(s).split(',')[-1]), dtype='float32')

        return np.concatenate([f, f_mask])


    def get_next_state(self, state, action):
        '''returns the next state given a state and an action'''
        action = int(action)
        if action == 4: # Stay action, do nothing
            return state

        a_mask = self.spec.agent_mask
        b_mask = self.spec.box_mask
        wall_mask = self.spec.wall_mask
        rows = b_mask.shape[0]
        cols = b_mask.shape[1]

        a_coord = np.where(state.a_pos)
        a_coord_new = copy(a_coord)
        b_coord_new = (-1, -1)

        # movement
        if action not in [0, 1, 2, 3]:
            raise ValueError("Invalid action {}".format(action))

        move_agent = shift_coord_array[action]
        move_agent = (move_agent[0] + a_coord[0], move_agent[1] + a_coord[1])

        # move_agent is not in the wall
        if move_agent[0]>=0 and move_agent[0]<rows and move_agent[1]>=0 and move_agent[1]<cols:
            if wall_mask[move_agent]==False:
                # if not moving into a box
                if state.b_pos[move_agent]==False:
                    a_coord_new = (move_agent[0], move_agent[1])
                else:
                    # tries to move box
                    move_box = shift_coord_array[action]
                    move_box = (move_box[0]*2 + a_coord[0], move_box[1]*2 + a_coord[1])
                    if move_box[0]>=0 and move_box[0]<rows and move_box[1]>=0 and move_box[1]<cols:
                        if wall_mask[move_box]==False and state.b_pos[move_box]==False:
                            a_coord_new = (move_agent[0], move_agent[1])
                            b_coord_new = (move_box[0], move_box[1])


        # update a_pos
        a_pos_new = np.zeros_like(state.a_pos)
        a_pos_new[a_coord_new] = True
        b_pos_new = deepcopy(state.b_pos)
        # update b_pos
        if b_coord_new != (-1, -1):
            b_pos_new[a_coord_new] = False
            b_pos_new[b_coord_new] = True

        return BoxesEnvState(a_pos_new, b_pos_new)


    def print_state(self, state):
        '''
        - Grey blocks are walls
        - Green blocks are boxes
        - The purple block is the goal state
        - The yellow block is the agent
        '''
        rows = state.b_pos.shape[0]
        cols = state.b_pos.shape[1]

        canvas = np.zeros(tuple([2*rows-1, 2*cols+1]), dtype='int8')

        # cell borders
        for i in range(1, canvas.shape[0], 2):
            canvas[i, :] = 1
        for j in range(0, canvas.shape[1], 2):
            canvas[:, j] = 2

        # agent
        for i in range(rows):
            for j in range(cols):
                if state.a_pos[i, j]==1:
                    canvas[2*i, 2*j+1] = 3

        # boxes
        for i in range(rows):
            for j in range(cols):
                if state.b_pos[i, j]==1:
                    canvas[2*i, 2*j+1] = 4
        # goal
        for i in range(rows):
            for j in range(cols):
                if spec.goal_mask[i, j]==1:
                    canvas[2*i, 2*j+1] = 5

        # walls
        for i in range(rows):
            for j in range(cols):
                if spec.wall_mask[i, j]==1:
                    canvas[2*i, 2*j+1] = 6

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
                    print('█'+black_color, end='')
            print('')
