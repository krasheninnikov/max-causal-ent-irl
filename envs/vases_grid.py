import numpy as np
from collections import defaultdict
from copy import copy, deepcopy
from envs.env import DeterministicEnv
from envs.utils import unique_perm, zeros_with_ones


class VasesEnvState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, v_pos, bv_pos, a_pos, t_pos, carrying):
        self.v_pos = v_pos
        self.bv_pos = bv_pos
        self.a_pos = a_pos
        self.t_pos = t_pos
        # Variable determining whether the agent is carrying something:
        # [0, 0] -> nothing, [1, 0] -> vase, [0, 1] -> tablecloth
        self.carrying = carrying

        self.v_pos_tuple = tuple(v_pos.flatten())
        self.bv_pos_tuple = tuple(bv_pos.flatten())
        self.a_pos_tuple = tuple(a_pos.flatten())
        self.t_pos_tuple = tuple(t_pos.flatten())
        self.carrying_tuple = tuple(carrying)

    def __eq__(self, other):
        return isinstance(other, VasesEnvState) and \
            self.v_pos_tuple == other.v_pos_tuple and \
            self.bv_pos_tuple == other.bv_pos_tuple and \
            self.a_pos_tuple == other.a_pos_tuple and \
            self.t_pos_tuple == other.t_pos_tuple and \
            self.carrying_tuple == other.carrying_tuple

    def __hash__(self):
        return hash(self.v_pos_tuple + self.bv_pos_tuple + self.a_pos_tuple + self.t_pos_tuple + self.carrying_tuple)


class VasesGrid(DeterministicEnv):
    def __init__(self, spec, f_include_masks=False, compute_transitions=True):
        self.spec = spec
        self.init_state = deepcopy(spec.init_state)

        self.nA = 6

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
        carrying = np.zeros(2, dtype='bool')
        n_v = self.spec.n_v
        n_t = _v = self.spec.n_t
        # Could be at the same location as the agent, or at any of the ds
        n_v_pos = 1 + np.sum(self.spec.d_mask)
        n_bv_pos = np.sum(self.spec.bv_mask)
        n_a_pos = np.sum(self.spec.agent_mask)
        n_t_pos = 1 + np.sum(self.spec.t_mask)
        state_num = {}

        # Possible agent positions
        for a_pos in unique_perm(zeros_with_ones(n_a_pos, 1)):
            agent_mask_pos = np.zeros_like(self.spec.agent_mask.flatten())
            np.put(agent_mask_pos, np.where(self.spec.agent_mask.flatten()), a_pos)
            agent_mask_pos = agent_mask_pos.reshape(self.spec.agent_mask.shape)

            # Possible vases and broken vases
            for n_bv in range(n_v+1):
                # n_places_for_vase choose n_vases
                for v_pos in unique_perm(zeros_with_ones(n_v_pos, n_v-n_bv)):

                    # Determine legal locations for the vase: it can be at desks,
                    # or at the agent's inventory. Otherwise it would
                    # have to be broken.

                    # Place n_v-n_bv vases into the legal pos in the mask
                    v_mask_pos = np.zeros_like(self.spec.d_mask.flatten())
                    np.put(v_mask_pos, np.where(self.spec.d_mask.flatten()), v_pos[:-1])
                    v_mask_pos = v_mask_pos.reshape(self.spec.d_mask.shape)

                    # last element of v_pos is the agent's inventory
                    carrying[0] = v_pos[-1]

                    # Possible broken vase positions
                    for bv_pos in unique_perm(zeros_with_ones(n_bv_pos, n_bv)):
                        bv_mask_pos = np.zeros_like(self.spec.bv_mask.flatten())
                        np.put(bv_mask_pos, np.where(self.spec.bv_mask.flatten()), bv_pos)
                        bv_mask_pos = bv_mask_pos.reshape(self.spec.d_mask.shape)

                        # Possible tablecloth positions
                        for t_pos in unique_perm(zeros_with_ones(n_t_pos, n_t)):

                            # last element of t_pos is the agent's inventory
                            carrying[1] = t_pos[-1]

                            # don't count states where the agent carries both a
                            # vase and a tablecloth since these aren't permitted
                            if np.sum(carrying)==2:
                                continue

                            t_mask_pos = np.zeros_like(self.spec.t_mask.flatten())
                            np.put(t_mask_pos, np.where(self.spec.t_mask.flatten()), t_pos[:-1])
                            t_mask_pos = t_mask_pos.reshape(self.spec.t_mask.shape)

                            state = VasesEnvState(v_mask_pos,
                                                  bv_mask_pos,
                                                  agent_mask_pos,
                                                  t_mask_pos,
                                                  deepcopy(carrying))

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
        - Number of broken vases
        - Number of vases on tables
        - Number of tablecloths on tables
        - Number of tablecloths on floors
        - Number of vases on desks
        - Number of tablecloths on desks
        '''
        if include_masks==None:
            include_masks = self.f_include_masks

        vases_on_tables = np.logical_and(s.v_pos, self.spec.table_mask)
        tablecloths_on_tables = np.logical_and(s.t_pos, self.spec.table_mask)
        f = np.asarray([np.sum(s.bv_pos), # number of broken vases
             np.sum(vases_on_tables), # number of vases on tables
             np.sum(tablecloths_on_tables),
             np.sum(np.logical_and(s.t_pos, self.spec.bv_mask)), # number of tablecloths on the floor
             np.sum(np.logical_xor(vases_on_tables, np.logical_and(s.v_pos, self.spec.d_mask))), # vases on desks
             np.sum(np.logical_xor(tablecloths_on_tables, np.logical_and(s.t_pos, self.spec.d_mask)))]) # tablecloths on desks

        f_mask = np.array([])
        if include_masks:
            f_mask = np.array(list(self.state_to_str(s).split(',')[-1]), dtype='float32')


        return np.concatenate([f, f_mask])


    def get_next_state(self, state, action):
        '''returns the next state given a state and an action'''
        action = int(action)
        new_v_pos, new_bv_pos, new_a_pos, new_t_pos, new_carrying = map(
            deepcopy,
            [state.v_pos, state.bv_pos, state.a_pos, state.t_pos, state.carrying])

        d_mask = self.spec.d_mask
        n, m = d_mask.shape

        a_coord = np.where(state.a_pos)
        a_coord_new = copy(a_coord)

        shift_coord_array = [[-1, 0], [0, 1], [1, 0], [0, -1]]

        # movement
        if action in [0, 1, 2, 3]:
            move_coord = shift_coord_array[action]
            move_coord = (move_coord[0] + a_coord[1], move_coord[1] + a_coord[2])

            # move_coord is not in the wall, and no desk at move_coord
            if move_coord[0]>=0 and move_coord[0]<n and move_coord[1]>=0 and move_coord[1]<m:
                if d_mask[move_coord]==False:
                    a_coord_new = (a_coord[0], move_coord[0], move_coord[1])

            # rotate to the correct position
            a_coord_new = (action, a_coord_new[1], a_coord_new[2])

            # update a_pos
            new_a_pos = np.zeros_like(state.a_pos)
            new_a_pos[a_coord_new] = True

        elif action == 4:
            # Stay action; do nothing
            pass

        elif action == 5:
            obj_coord = shift_coord_array[int(a_coord[0])]
            obj_coord = (obj_coord[0] + a_coord[1], obj_coord[1] + a_coord[2])

            # no wall at obj_coord
            if obj_coord[0]>=0 and obj_coord[0]<n and obj_coord[1]>=0 and obj_coord[1]<m:
                # Try to pick up an object
                if state.carrying[0] + state.carrying[1] == 0:
                    # vase
                    if state.v_pos[obj_coord] == True:
                        new_v_pos[obj_coord] = False
                        new_carrying[0] = 1

                    # tablecloth
                    elif state.t_pos[obj_coord] == True:
                        new_t_pos[obj_coord] = False
                        new_carrying[1] = 1

                # Try to put down an object
                else:
                    # carrying a vase and there's no other vase at obj_coord
                    if state.carrying[0] == 1 and state.v_pos[obj_coord] == False:
                        # vase doesn't break
                        if self.spec.d_mask[obj_coord]:
                            new_v_pos[obj_coord] = True
                            new_carrying[0] = 0

                        # vase breaks
                        elif self.spec.bv_mask[obj_coord]:
                            # not allowing two broken vases in one spot
                            if state.bv_pos[obj_coord] == False:
                                new_bv_pos[obj_coord] = True
                                new_carrying[0] = 0

                    # carrying a tablecloth
                    if state.carrying[1] == 1 and self.spec.t_mask[obj_coord]:
                        # cannot put into a cell already containing tablecloth or
                        # a vase (tablecloths go under vases)
                        if not state.t_pos[obj_coord] and not state.v_pos[obj_coord]:
                            new_t_pos[obj_coord] = True
                            new_carrying[1] = 0

        return VasesEnvState(new_v_pos, new_bv_pos, new_a_pos, new_t_pos, new_carrying)


    def print_state(self, state):
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
        n, m = self.spec.d_mask.shape

        canvas = np.zeros(tuple([3*n-1, 3*m+1]), dtype='int8')

        # cell borders
        for i in range(2, canvas.shape[0], 3):
            canvas[i, :] = 1
        for j in range(0, canvas.shape[1], 3):
            canvas[:, j] = 2

        # desks
        for i in range(n):
            for j in range(m):
                if self.spec.d_mask[i, j]==1:
                    canvas[3*i+1, 3*j+1] = 3

        # vases
        for i in range(n):
            for j in range(m):
                if state.v_pos[i, j]==1:
                    canvas[3*i, 3*j+2] = 4

        # tablecloths
        for i in range(n):
            for j in range(m):
                if state.t_pos[i, j]==1:
                    canvas[3*i, 3*j+1] = 5

        # broken vases
        for i in range(n):
            for j in range(m):
                if state.bv_pos[i, j]==1:
                    canvas[3*i+1, 3*j+2] = 6

        # agent
        for rotation in range(4):
            for i in range(n):
                for j in range(m):
                    if state.a_pos[rotation, i, j]==1:
                        canvas[3*i+1, 3*j+1] = 7+rotation

        # tables; it's important for this for loop to be after the d_mask for loop
        # since table_mask is in d_mask
        for i in range(n):
            for j in range(m):
                if self.spec.table_mask[i, j]==1:
                    canvas[3*i+1, 3*j+1] = 11

        black_color = '\x1b[0m'
        purple_background_color = '\x1b[0;35;85m'

        agent_color = black_color
        if state.carrying[0] == 1:
            agent_color = '\x1b[1;42;42m'
        if state.carrying[1] == 1:
            agent_color = '\x1b[1;45;45m'

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
                elif char_num==7:
                    print(agent_color+'↑'+black_color, end='')
                elif char_num==8:
                    print(agent_color+'→'+black_color, end='')
                elif char_num==9:
                    print(agent_color+'↓'+black_color, end='')
                elif char_num==10:
                    print(agent_color+'←'+black_color, end='')
                elif char_num==11:
                    print('\033[93m█'+black_color, end='')
            print('')
