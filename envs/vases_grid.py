import numpy as np
from collections import defaultdict
from copy import copy, deepcopy
from gym import spaces
from envs.utils import unique_perm, zeros_with_ones


class VasesEnvState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, d_pos, v_pos, bv_pos, a_pos, t_pos, carrying, table_pos):
        self.d_pos = d_pos
        self.v_pos = v_pos
        self.bv_pos = bv_pos
        self.a_pos = a_pos
        self.t_pos = t_pos
        # Variable determining whether the agent is carrying something:
        # [0, 0] -> nothing, [1, 0] -> vase, [0, 1] -> tablecloth
        self.carrying = carrying
        self.table_pos = table_pos


class VasesGrid(object):
    def __init__(self, spec, init_state):
        self.nA = 5
        self.action_space = spaces.Discrete(self.nA)

        self.r_vec = np.array([0,0,1,0,0,0], dtype='float32')
        self.observation_space = spaces.Box(low=0, high=3, shape=self.r_vec.shape, dtype=np.float32)

        self.timestep = 0

        self.spec = spec
        self.init_state = deepcopy(init_state)
        self.s = deepcopy(init_state)

        self.enumerate_states()
        self.make_f_matrix()
        self.get_deterministic_transitions()

    def enumerate_states(self):
        carrying = np.zeros(2, dtype='bool')
        n_v = self.spec.n_v
        n_t = _v = self.spec.n_t
        # Could be at the same location as the agent, or at any of the ds
        n_v_pos = 1 + np.sum(self.spec.d_mask)
        n_bv_pos = np.sum(self.spec.bv_mask)
        n_a_pos = np.sum(self.spec.agent_mask)
        n_t_pos = 1 + np.sum(self.spec.t_mask)
        P = {}
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

                            state = VasesEnvState(self.spec.d_mask,
                                                  v_mask_pos,
                                                  bv_mask_pos,
                                                  agent_mask_pos,
                                                  t_mask_pos,
                                                  carrying,
                                                  self.spec.table_mask)
                            state_str = state_to_str(state)

                            if state_str not in state_num:
                                state_num[state_str] = len(state_num)

        # Take every possible action from each of the possible states. Since the
        # env is deterministic, this is sufficient to get transition probs
        for state_str, state_num_id in state_num.items():
            P[state_num_id] = {}
            for action in range(5):
                statep = self.state_step(action, str_to_state(state_str))
                statep_str = state_to_str(statep)
                P[state_num_id][action] = [(1, state_num[statep_str], 0)]

        self.state_num = state_num
        self.num_state = {v: k for k, v in self.state_num.items()}
        self.P = P
        self.nS = len(P.keys())


    def get_transition_matrix(self):
        '''Create self.T, a matrix with index S,A,S' -> P(S'|S,A)      '''
        self.T = np.zeros([self.nS, self.nA, self.nS])
        for s in range(self.nS):
            for a in range(self.nA):
                self.T[s, a, self.P[s][a][0][1]] = 1


    def get_deterministic_transitions(self):
        '''Create self.deterministic_T, a matrix with index S,A -> S'   '''
        self.deterministic_T = np.zeros((self.nS, self.nA), dtype='int32')
        for s in range(self.nS):
            for a in range(self.nA):
                self.deterministic_T[s,a]=self.P[s][a][0][1]


    def make_f_matrix(self):
         self.f_matrix = np.zeros((self.nS, 6))
         for state_str, state_num_id in self.state_num.items():
             self.f_matrix[state_num_id, :] = self.s_to_f(str_to_state(state_str))


    def s_to_f(self, s):
        '''
        returns features of the state:
        - Number of broken vases
        - Number of vases on tables
        - Number of tablecloths on tables
        - Number of tablecloths on floors
        - Number of vases on desks
        - Number of tablecloths on desks
        '''
        vases_on_tables = np.logical_and(s.v_pos, self.spec.table_mask)
        tablecloths_on_tables = np.logical_and(s.t_pos, self.spec.table_mask)
        f = np.asarray([np.sum(s.bv_pos), # number of broken vases
             np.sum(vases_on_tables), # number of vases on tables
             np.sum(tablecloths_on_tables),
             np.sum(np.logical_and(s.t_pos, self.spec.bv_mask)), # number of tablecloths on the floor
             np.sum(np.logical_xor(vases_on_tables, np.logical_and(s.v_pos, s.d_pos))), # vases on desks
             np.sum(np.logical_xor(tablecloths_on_tables, np.logical_and(s.t_pos, s.d_pos)))]) # tablecloths on desks
        return f


    def reset(self):
        self.timestep = 0
        self.s = deepcopy(self.init_state)


    def state_step(self, action, state=None):
        '''returns the next state given a state and an action'''
        action = int(action)

        if state==None: state = self.s
        d_mask = self.spec.d_mask
        n = d_mask.shape[0]
        m = d_mask.shape[1]

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
            a_pos_new = np.zeros_like(state.a_pos)
            a_pos_new[a_coord_new] = True
            state.a_pos = a_pos_new

        elif action==4:
            obj_coord = shift_coord_array[int(a_coord[0])]
            obj_coord = (obj_coord[0] + a_coord[1], obj_coord[1] + a_coord[2])

            # no wall at obj_coord
            if obj_coord[0]>=0 and obj_coord[0]<n and obj_coord[1]>=0 and obj_coord[1]<m:
                # Try to pick up an object
                if state.carrying[0] + state.carrying[1] == 0:
                    # vase
                    if state.v_pos[obj_coord] == True:
                        state.v_pos[obj_coord] = False
                        state.carrying[0] = 1

                    # tablecloth
                    elif state.t_pos[obj_coord] == True:
                        state.t_pos[obj_coord] = False
                        state.carrying[1] = 1

                # Try to put down an object
                else:
                    # carrying a vase and there's no other vase at obj_coord
                    if state.carrying[0] == 1 and state.v_pos[obj_coord] == False:
                        # vase doesn't break
                        if self.spec.d_mask[obj_coord]:
                            state.v_pos[obj_coord] = True
                            state.carrying[0] = 0

                        # vase breaks
                        elif self.spec.bv_mask[obj_coord]:
                            # not allowing two broken vases in one spot
                            if state.bv_pos[obj_coord] == False:
                                state.bv_pos[obj_coord] = True
                                state.carrying[0] = 0

                    # carrying a tablecloth
                    if state.carrying[1] == 1 and self.spec.t_mask[obj_coord]:
                        # cannot put into a cell already containing tablecloth or
                        # a vase (tablecloths go under vases)
                        if not state.t_pos[obj_coord] and not state.v_pos[obj_coord]:
                            state.t_pos[obj_coord] = True
                            state.carrying[1] = 0

        return state


    def step(self, action):
        '''
        given an action, takes a step from self.s, updates self.s and returns:
        - the observation (features of the next state)
        - the associated reward
        - done, the indicator of completed episode
        - info
        '''
        self.state_step(action)
        obs = self.s_to_f(self.s)

        self.timestep+=1
        done = False
        if self.timestep>50: done=True

        info = defaultdict(lambda : '')
        done = bool(done)
        return np.array([obs]), np.array([obs.T @ self.r_vec]), np.array([done], dtype='bool'), [info]


    def close(self):
        self.reset()


    def seed(self, seed=None):
        pass


def state_to_str(state):
    '''
    returns a string encoding of a state to serve as key in the state dictionary
    '''
    string = str(state.d_pos.shape[0]) + "," + str(state.d_pos.shape[1]) + ","
    string += np.array_str(state.d_pos.flatten().astype(int))[1:-1]
    string += np.array_str(state.v_pos.flatten().astype(int))[1:-1]
    string += np.array_str(state.bv_pos.flatten().astype(int))[1:-1]
    string += np.array_str(state.t_pos.flatten().astype(int))[1:-1]
    string += np.array_str(state.table_pos.flatten().astype(int))[1:-1]
    string += np.array_str(state.a_pos.flatten().astype(int))[1:-1]
    string += np.array_str(np.asarray(state.carrying).astype(int))[1:-1]

    return string.replace(" ", "")


def str_to_state(string):
    '''
    returns a state from a string encoding
    assumes states are represented as binary masks
    '''
    rpos = string.find(",")
    rows = int(string[:rpos])
    string = string[rpos+1:]

    cpos = string.find(",")
    cols = int(string[:cpos])
    string = string[cpos+1:]

    d_pos = np.asarray(list(string[:rows*cols]))
    d_pos = (d_pos > '0').reshape(rows, cols)
    string = string[rows*cols:]

    v_pos = np.asarray(list(string[:rows*cols]))
    v_pos = (v_pos > '0').reshape(rows, cols)
    string = string[rows*cols:]

    bv_pos = np.asarray(list(string[:rows*cols]))
    bv_pos = (bv_pos > '0').reshape(rows, cols)
    string = string[rows*cols:]

    t_pos = np.asarray(list(string[:rows*cols]))
    t_pos = (t_pos > '0').reshape(rows, cols)
    string = string[rows*cols:]

    table_pos = np.asarray(list(string[:rows*cols]))
    table_pos = (table_pos > '0').reshape(rows, cols)
    string = string[rows*cols:]

    a_pos = np.asarray(list(string[:4*rows*cols]))
    a_pos = (a_pos > '0').reshape(4, rows, cols)
    string = string[4*rows*cols:]

    carrying = [int(string[0]), int(string[1])]

    return VasesEnvState(d_pos, v_pos, bv_pos, a_pos, t_pos, carrying, table_pos)


def print_state(state):
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
    n = state.d_pos.shape[0]
    m = state.d_pos.shape[1]

    canvas = np.zeros(tuple([3*n-1, 3*m+1]), dtype='int8')

    # cell borders
    for i in range(2, canvas.shape[0], 3):
        canvas[i, :] = 1
    for j in range(0, canvas.shape[1], 3):
        canvas[:, j] = 2

    # desks
    for i in range(n):
        for j in range(m):
            if state.d_pos[i, j]==1:
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

    # tables; it's important for this for loop to be after the d_pos for loop
    # since table_pos is in d_pos
    for i in range(n):
        for j in range(m):
            if state.table_pos[i, j]==1:
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
