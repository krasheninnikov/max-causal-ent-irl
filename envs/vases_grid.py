import numpy as np
import operator as op
from scipy.special import comb
from copy import copy, deepcopy
from utils import unique_perm, zeros_with_ones

class VasesEnvSpec(object):
    def __init__(self, n_v, n_t, d_mask, v_mask, bv_mask, agent_mask, t_mask):
        self.n_v = n_v # number of vases
        self.n_t = n_t # number of tablecloths
        self.d_mask = d_mask # desk location
        self.v_mask = v_mask # places where vases can be
        self.bv_mask = bv_mask # places where broken vases can be
        self.agent_mask = agent_mask # places where agent can be
        self.t_mask = t_mask # places where tablecloths can be


class VasesEnvState(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self, d_pos, v_pos, bv_pos, a_pos, t_pos, carrying):
        self.d_pos = d_pos
        self.v_pos = v_pos
        self.bv_pos = bv_pos
        self.a_pos = a_pos
        self.t_pos = t_pos
        # Variable determining whether the agent is carrying something:
        # [0, 0] -> nothing, [1, 0] -> vase, [0, 1] -> tablecloth
        self.carrying = carrying


def print_state(state):
    '''
    Renders the state. Each tile in the gridworld corresponds to a 2x2 cell in
    the rendered state.
    - Green tiles correspond to vases
    - Red tiles correspond to broken vases
    - Brown tiles correspond to tables
    - Purple tiles correspond to tablecloths
    - The arrow and its direction correspond to the agent and its rotation. The
      color of the arrow corres to the object the agent is carrying. The agent is
      rendered in the same subcell as tables are since the agent and the table
      are never in the same cell.
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
                canvas[2*i+i+1, 2*j+j+1] = 3

    # vases
    for i in range(n):
        for j in range(m):
            if state.v_pos[i, j]==1:
                canvas[2*i+i, 2*j+j+2] = 4

    # tablecloths
    for i in range(n):
        for j in range(m):
            if state.t_pos[i, j]==1:
                canvas[2*i+i, 2*j+j+1] = 5

    # broken vases
    for i in range(n):
        for j in range(m):
            if state.bv_pos[i, j]==1:
                canvas[2*i+i+1, 2*j+j+2] = 6

    # agent
    for rotation in range(4):
        for i in range(n):
            for j in range(m):
                if state.a_pos[rotation, i, j]==1:
                    canvas[2*i+i+1, 2*j+j+1] = 7+rotation

    black_color = '\033[0m'
    agent_color = black_color
    if state.carrying[0] == 1:
        agent_color = '\033[92m'
    if state.carrying[1] == 1:
        agent_color = '\033[95m'

    for line in canvas:
        for char_num in line:
            if char_num==0:
                print('\u2003', end='')
            elif char_num==1:
                print('\u2013', end='')
            elif char_num==2:
                print('|', end='')
            elif char_num==3:
                print('\033[93m█\033[0m', end='')
            elif char_num==4:
                print('\033[92m█\033[0m' , end='')
            elif char_num==5:
                print('\033[95m█\033[0m', end='')
            elif char_num==6:
                print('\033[91m█\033[0m', end='')

            elif char_num==7:
                print(agent_color+'↑'+black_color, end='')
            elif char_num==8:
                print(agent_color+'→'+black_color, end='')
            elif char_num==9:
                print(agent_color+'↓'+black_color, end='')
            elif char_num==10:
                print(agent_color+'←'+black_color, end='')
        print('')


class VasesGrid(object):
    def __init__(self, spec, init_state):
        self.spec = spec
        self.init_state = deepcopy(init_state)
        self.s = deepcopy(init_state)

        # testing the functions below, they shouldn't be here in the final env
        self.enumerate_states()
        self.step(self.s, 0)

    def enumerate_states(self):
        i = 0
        carrying = np.zeros(2)j
        n_v = self.spec.n_v
        n_t = _v = self.spec.n_t
        # Could be at the same location as the agent, or at any of the ds
        n_v_pos = 1 + np.sum(self.spec.d_mask)
        n_bv_pos = np.sum(self.spec.bv_mask)
        n_a_pos = np.sum(self.spec.agent_mask)
        n_t_pos = 1 + np.sum(self.spec.t_mask)
        self.P = {}

        # Possible agent positions
        for a_pos in unique_perm(zeros_with_ones(n_a_pos, 1)):
            agent_mask_pos = np.zeros_like(self.spec.agent_mask.flatten())
            np.put(agent_mask_pos, np.where(self.spec.agent_mask.flatten()), a_pos)
            agent_mask_pos = agent_mask_pos.reshape(self.spec.agent_mask.shape)

            # Possible vases and broken vases
            for n_bv in range(n_v):
                # n_places_for_vase choose n_vases
                for v_pos in unique_perm(zeros_with_ones(n_v_pos, n_v-n_bv)):

                    # Determine legal locations for the vase: it can be at desks,
                    # or at the agent's inventory. Otherwise it would
                    # have to be broken.

                    # Place n_v-n_bv vases into the legal pos in the mask
                    v_mask_pos = np.zeros_like(self.spec.d_mask.flatten())
                    np.put(v_mask_pos, np.where(self.spec.d_mask.flatten()), v_pos[:-1])
                    v_mask_pos = v_mask_pos.reshape(self.spec.v_mask.shape)

                    # last element of v_pos is the agent's inventory
                    carrying[0] = v_pos[-1]

                    # Possible broken vase positions
                    for bv_pos in unique_perm(zeros_with_ones(n_bv_pos, n_bv)):
                        bv_mask_pos = np.zeros_like(self.spec.bv_mask.flatten())
                        np.put(bv_mask_pos, np.where(self.spec.bv_mask.flatten()), v_pos)

                        # Possible tablecloth positions
                        for t_pos in unique_perm(zeros_with_ones(n_t_pos, n_t)):
                            # exclude states where the agent carries both the vase and the tablecloth
                            if t_pos[-1]==1 and carrying[0]==1:
                                break

                            # last element of t_pos is the agent's inventory
                            t_mask_pos = np.zeros_like(self.spec.t_mask.flatten())
                            np.put(t_mask_pos, np.where(self.spec.t_mask.flatten()), t_pos[:-1])
                            t_mask_pos = t_mask_pos.reshape(self.spec.t_mask.shape)
                            carrying[1] = t_pos[-1]

                            # TODO add the state to P; how to best store states?

    def reset(self):
        self.s = deepcopy(self.init_state)

    def step(self, state, action):
        'returns the next state given a state and an action'

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
                        if self.spec.v_mask[obj_coord]:
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
