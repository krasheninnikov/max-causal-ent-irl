import numpy as np
from collections import defaultdict
from copy import copy, deepcopy
from gym import spaces
from envs.utils import unique_perm, zeros_with_ones

shift_coord_array = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class BoxesEnvState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, a_pos, b_pos):
        self.a_pos = np.array(a_pos, dtype='bool')
        self.b_pos = np.array(b_pos, dtype='bool')


class BoxesEnv(object):
    def __init__(self, spec, init_state, f_include_masks=False, compute_transitions=True):
        self.spec = spec

        self.init_state = deepcopy(init_state)
        self.s = deepcopy(init_state)

        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)

        self.nF = 4
        self.f_include_masks = f_include_masks
        f_len = len(self.s_to_f(init_state))

        # TODO: Figure out if you want to change the way the reward works to the original environment
        self.r_vec = np.concatenate([np.array([0,0,0,1], dtype='float32'),
                                     np.zeros(f_len-self.nF, dtype='float32')])
        self.observation_space = spaces.Box(low=0, high=255, shape=self.r_vec.shape, dtype=np.float32)

        self.timestep = 0

        if compute_transitions:
            self.enumerate_states()
            self.make_f_matrix()
            self.get_deterministic_transitions()
            self.get_deterministic_transitions_transpose()

    def enumerate_states(self):
        P = {}
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
                    state_str = self.state_to_str(state)
                    # print("Enumerating")
                    # print(state.a_pos.astype(int))
                    # print(state.b_pos.astype(int))
                    # print()
                    # import pdb; pdb.set_trace()

                    if state_str not in state_num:
                        state_num[state_str] = len(state_num)

        # Take every possible action from each of the possible states. Since the
        # env is deterministic, this is sufficient to get transition probs
        for state_str, state_num_id in state_num.items():
            P[state_num_id] = {}
            for action in range(self.nA):
                statep = self.state_step(action, self.str_to_state(state_str))
                # print(action)
                # print(str_to_state(state_str).a_pos.astype(int))
                # print(str_to_state(state_str).b_pos.astype(int))
                # print()
                # print(statep.a_pos.astype(int))
                # print(statep.b_pos.astype(int))
                # print()
                # print()
                statep_str = self.state_to_str(statep)
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
                
    def get_deterministic_transitions_transpose(self):
        '''Create self.deterministic_transpose, a matrix with index S,A -> S', with the inverse dynamics '''
        self.deterministic_transpose = np.zeros((self.nS, self.nA), dtype='int32')
        for s in range(self.nS):
            for a in range(self.nA):
                self.deterministic_transpose[self.P[s][a][0][1],a]=s


    def make_f_matrix(self):
         self.f_matrix = np.zeros((self.nS, self.nF))
         for state_str, state_num_id in self.state_num.items():
             self.f_matrix[state_num_id, :] = self.s_to_f(self.self.str_to_state(state_str))


    def s_to_f(self, s, include_masks=None):
        '''
        returns features of the state:
        - Number of boxes adjacent to zero walls
        - Number of boxes adjacent to one wall
        - Number of boxes adjacent to two or more walls
        - Whether agent is on the goal state or not
        '''
        if include_masks==None:
            include_masks = self.f_include_masks
        f = np.zeros(self.nF)

        # Iterate over box positions 
        for r in range(s.b_pos.shape[0]):
            for c in range(s.b_pos.shape[1]):
                # Count adjacent boxes
                surround = 0
                for shift in shift_coord_array:
                    if r+shift[0] < len(s.b_pos) and r+shift[0] >= 0 and c+shift[1] < len(s.b_pos[r+shift[0]]) and c+shift[1] >= 0:
                        if self.spec.wall_mask[r+shift[0]][c+shift[1]]:
                            surround += 1
                f[min(surround, 2)] += 1
        f[-1] = np.sum(np.logical_and(s.a_pos, self.spec.goal_mask))

        f_mask = np.array([])
        if include_masks:
            f_mask = np.array(list(self.state_to_str(s).split(',')[-1]), dtype='float32')

        return np.concatenate([f, f_mask])


    def reset(self):
        self.timestep = 0
        self.s = deepcopy(self.init_state)

        obs = self.s_to_f(self.s)
        return np.array(obs, dtype='float32').flatten() #, obs.T @ self.r_vec, False, defaultdict(lambda : '')


    def state_step(self, action, state=None):
        '''returns the next state given a state and an action'''
        action = int(action)

        if state==None: state = self.s

        a_mask = self.spec.agent_mask
        b_mask = self.spec.box_mask
        wall_mask = self.spec.wall_mask
        rows = b_mask.shape[0]
        cols = b_mask.shape[1]

        a_coord = np.where(state.a_pos)
        a_coord_new = copy(a_coord)
        b_coord_new = (-1, -1)

        # movement
        if action in [0, 1, 2, 3]:
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
                            if wall_mask[move_box]==False:
                                a_coord_new = (move_agent[0], move_agent[1])
                                b_coord_new = (move_box[0], move_box[1])


            # update a_pos
            a_pos_new = np.zeros_like(state.a_pos)
            a_pos_new[a_coord_new] = True
            state.a_pos = a_pos_new
            # update b_pos
            if b_coord_new != (-1, -1):
                state.b_pos[a_coord_new] = False
                state.b_pos[b_coord_new] = True

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


    def reset(self):
        self.timestep = 0
        self.s = deepcopy(self.init_state)

        obs = self.s_to_f(self.s)
        return np.array([obs], dtype='float32').flatten() #, obs.T @ self.r_vec, False, defaultdict(lambda : '')

    def print_state(self, state, spec):
        '''
        TODO: Describe appearance of printed state
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
        for i in range(n):
            for j in range(m):
                if state.a_pos[i, j]==1:
                    canvas[2*i+1, 2*j+1] = 3

        # boxes
        for i in range(n):
            for j in range(m):
                if state.b_pos[i, j]==1:
                    canvas[2*i+1, 2*j+1] = 4
        # goal
        for i in range(n):
            for j in range(m):
                if spec.goal_mask[i, j]==1:
                    canvas[2*i, 2*j+2] = 5

        # walls
        for i in range(n):
            for j in range(m):
                if spec.wall_mask[i, j]==1:
                    canvas[2*i, 2*j+2] = 6

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


    def state_to_str(self, state):
        '''
        returns a string encoding of a state to serve as key in the state dictionary
        '''
        string = str(state.a_pos.shape[0]) + "," + str(state.a_pos.shape[1]) + ","
        string += np.array_str(state.a_pos.flatten().astype(int))
        string += np.array_str(state.b_pos.flatten().astype(int))
        return string


    def str_to_state(self, string):
        '''
        returns a state from a string encoding
        assumes states are represented as binary masks
        '''
        cpos = string.find(",")
        rows = int(string[:cpos])
        string = string[cpos+1:]

        cpos = string.find(",")
        cols = int(string[:cpos])
        string = string[cpos+1:]

        cpos = string.find("]")
        a_pos = string[1:cpos].split(" ")
        a_pos = np.array(a_pos).reshape(rows, cols)
        string = string[cpos+1:]

        cpos = string.find("]")
        b_pos = string[1:cpos].split(" ")
        b_pos = np.array(b_pos).reshape(rows, cols)

        return BoxesEnvState(a_pos, b_pos)