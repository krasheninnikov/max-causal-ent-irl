import numpy as np
import operator as op
from scipy.special import comb
from copy import copy
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
        # 0 -> nothing, 1 -> vase, 2 -> tablecloth
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
        canvas[:, j] = -1

    # desks
    for i in range(n):
        for j in range(m):
            if state.d_pos[i, j]==1:
                canvas[2*i+i+1, 2*j+j+1] = 2

    # vases
    for i in range(n):
        for j in range(m):
            if state.v_pos[i, j]==1:
                canvas[2*i+i, 2*j+j+2] = 3

    # tablecloths
    for i in range(n):
        for j in range(m):
            if state.v_pos[i, j]==1:
                canvas[2*i+i, 2*j+j+1] = 4

    # broken vases
    for i in range(n):
        for j in range(m):
            if state.bv_pos[i, j]==1:
                canvas[2*i+i, 2*j+j] = 5

    # agent
    for rotation in range(4):
        for i in range(n):
            for j in range(m):
                if state.a_pos[rotation, i, j]==1:
                    canvas[2*i+i+1, 2*j+j+1] = 6+rotation

    black_color = '\033[0m'
    agent_color = black_color
    if state.carrying==1:
        agent_color = '\033[92m'
    if state.carrying==2:
        agent_color = '\033[95m'

    for line in canvas:
        for char_num in line:
            if char_num==0:
                print('\u2003', end='')
            elif char_num==-1:
                print('|', end='')
            elif char_num==1:
                print('\u2013', end='')
            elif char_num==2:
                print('\033[93m█\033[0m', end='')
            elif char_num==3:
                print('\033[92m█\033[0m' , end='')
            elif char_num==4:
                print('\033[95m█\033[0m', end='')
            elif char_num==5:
                print('\033[91m█\033[0m', end='')

            elif char_num==6:
                print(agent_color+'↑'+black_color, end='')
            elif char_num==7:
                print(agent_color+'→'+black_color, end='')
            elif char_num==8:
                print(agent_color+'↓'+black_color, end='')
            elif char_num==9:
                print(agent_color+'←'+black_color, end='')
        print('')


class VasesGrid(object):
    def __init__(self, spec, init_state):
        self.spec = spec
        self.init_state = copy(init_state)
        self.s = copy(init_state)

        # testing the functions below, they shouldn't be here in the final env
        self.enumerate_states()
        self.step(self.s, 0)

    def enumerate_states(self):
        # TODO Jordan -> Dmitrii Comment: Ah, ok, I misunderstood what you were
        # saying in the call about coordinate lists vs. masks. Masks still seems
        # like the correct choice, though, since (size)^vases < prod i=0->vases (size-i) < (size choose vases)
        i = 0
        n_v = self.spec.n_v
        n_t = _v = self.spec.n_t
        # Could be at the same location as the agent, or at any of the ds
        n_v_pos = 1 + np.sum(self.spec.d_mask)
        n_bv_pos = np.sum(self.spec.bv_mask)
        n_a_pos = np.sum(self.spec.agent_mask)
        n_t_pos = np.sum(self.spec.t_mask)
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

                    # Determine legal locations for the vase: it can be at ds,
                    # or at the same location as the agent. Otherwise it would
                    # have to be broken. The sum is over agent rotations.
                    v_mask_legal = self.spec.d_mask + np.sum(agent_mask_pos, axis=0)

                    # Place n_v-n_bv vases into the legal pos in the mask
                    v_mask_pos = np.zeros_like(v_mask_legal.flatten())
                    np.put(v_mask_pos, np.where(v_mask_legal.flatten()), v_pos)
                    v_mask_pos = v_mask_pos.reshape(self.spec.v_mask.shape)

                    # Possible broken vase positions
                    # TODO Jordan -> Dmitrii Comment: Can we just keep track of
                    # the number broken vases instead? Less states, and since we
                    # can't pick them up anymore, I don't think the position matters much.

                    for bv_pos in unique_perm(zeros_with_ones(n_bv_pos, n_bv)):
                        bv_mask_pos = np.zeros_like(self.spec.bv_mask.flatten())
                        np.put(bv_mask_pos, np.where(self.spec.bv_mask.flatten()), v_pos)

                        # Possible tablecloth positions; the agent can carry
                        # either a vase or a tablecloth, not both. However,
                        # this is only reflected in the step function since it's
                        # possible for an agent carrying a vase to step on a tile
                        # that contains a dropped tablecloth

                        # TODO Jordan -> Dmitrii Comment: Do we want it to be possible
                        # for the agent to step on a broken vase? I'd lean on yes just
                        # because it seems easier to deal with.

                        for t_pos in unique_perm(zeros_with_ones(n_t_pos, n_t)):
                            t_mask_pos = np.zeros_like(self.spec.t_mask.flatten())
                            np.put(t_mask_pos, np.where(self.spec.t_mask.flatten()), t_pos)
                            t_mask_pos = t_mask_pos.reshape(self.spec.t_mask.shape)

                            # print(i); i+=1
                            # TODO enumerate over carrying object / object being
                            # on the floor
                            # TODO add the state to P; how to best store states?

    def reset(self):
        self.s = copy(self.init_state)

    def step(self, state, action):
        d_mask = self.spec.d_mask
        n = d_mask.shape[0]
        m = d_mask.shape[1]

        'returns the next state given a state and an action'
        a_coord = np.where(state.a_pos)
        a_coord_new = copy(a_coord)
        # movement
        if action in [0, 1, 2, 3]:
            # move up
            if action==0:
                # no wall and no desk above
                if a_coord[1]!=0 and d_mask[a_coord[1]-1, a_coord[2]]==0:
                    a_coord_new = tuple(map(op.add, a_coord, (0, -1, 0)))

            # moving right
            elif action==1:
                # no wall and no desk to the right
                if a_coord[2]!=m and d_mask[a_coord[1], a_coord[2]+1]==0:
                    a_coord_new = tuple(map(op.add, a_coord, (0, 0, 1)))

            # moving down
            elif action==2:
                # no wall and no desk below
                if a_coord[1]!=n and d_mask[a_coord[1]+1, a_coord[2]]==0:
                    a_coord_new = tuple(map(op.add, a_coord, (0, 1, 0)))

            # moving left
            elif action==3:
                # no wall and no desk to the left
                if a_coord[2]!=0 and d_mask[a_coord[1], a_coord[2]-1]==0:
                    a_coord_new = tuple(map(op.add, a_coord, (0, 0, -1)))

            # rotate to the correct position
            a_coord_new = (action, a_coord_new[1], a_coord_new[2])

            # update a_pos
            a_pos_new = np.zeros_like(state.a_pos)
            a_pos_new[a_coord_new] = True
            state.a_pos = a_pos_new

            # carrying an object
            if state.carrying==1:
                # update position of the vase that was at a_coord
                state.v_pos[a_coord[1], a_coord[2]]=False
                state.v_pos[a_coord_new[1], a_coord_new[2]]=True
            if state.carrying==2:
                # update position of the tablecloth that was at a_coord
                state.t_pos[a_coord[1], a_coord[2]]=False
                state.t_pos[a_coord_new[1], a_coord_new[2]]=True

        #TODO Jordan -> Dmitrii Comment: I ended up sticking with lots of ifs
        # rather than for loops bc consistency and no need to reatroactively
        # change code that works, but it ended up being lots of nested ifs.
        # I could declarte something like an array of tuples [(0, -1), (0, +1), (-1, 0), (+1, 0)]
        # and iterate over that instead, but currently planning on sticking with this.

        # pick/put object
        if action==4:
            # try to pick an object
            if state.carrying  == 0:
                # picking above
                if a_coord[0] == 0:

                    # vase above
                    if state.v_pos[a_coord[1] - 1, a_coord[2]] == True:
                        state.v_pos[a_coord[1] - 1, a_coord[2]] = False
                        state.carrying = 1

                    # tablecloth above
                    if state.t_pos[a_coord[1] - 1, a_coord[2]] == True:
                        state.t_pos[a_coord[1] - 1, a_coord[2]] = False
                        state.carrying = 2

                # picking right
                if a_coord[0] == 1:

                    # vase right
                    if state.v_pos[a_coord[1] , a_coord[2] + 1] == True:
                        state.v_pos[a_coord[1] , a_coord[2] + 1] = False
                        state.carrying = 1

                    # tablecloth right
                    if state.t_pos[a_coord[1], a_coord[2] + 1] == True:
                        state.t_pos[a_coord[1], a_coord[2] + 1] = False
                        state.carrying = 2

                # picking down
                if a_coord[0] == 2:

                    # vase down
                    if state.v_pos[a_coord[1] + 1, a_coord[2]] == True:
                        state.v_pos[a_coord[1] + 1, a_coord[2]] = False
                        state.carrying = 1

                    # tablecloth down
                    if state.t_pos[a_coord[1] + 1, a_coord[2]] == True:
                        state.t_pos[a_coord[1] + 1, a_coord[2]] = False
                        state.carrying = 2

                # picking left
                if a_coord[0] == 3:

                    # vase left
                    if state.v_pos[a_coord[1], a_coord[2] - 1] == True:
                        state.v_pos[a_coord[1], a_coord[2] - 1] = False
                        state.carrying = 1

                    # tablecloth left
                    if state.t_pos[a_coord[1], a_coord[2] - 1] == True:
                        state.t_pos[a_coord[1], a_coord[2] - 1] = False
                        state.carrying = 2

            # try to put an object
            else:
                # putting above
                if a_coord[0] == 0:

                    # vase above
                    if state.carrying == 1:

                        # vase doesn't break
                        if state.v_mask[a_coord[1] - 1, a_coord[2]]:
                            state.v_pos[a_coord[1] - 1, a_coord[2]] = True
                            state.carrying = 0

                        # vase breaks
                        elif state.bv_mask[a_coord[1] - 1, a_coord[2]]:
                            state.bv_pos[a_coord[1] - 1, a_coord[2]] = True
                            state.carrying = 0

                    # tablecloth above
                    if state.carrying == 2 and state.t_mask[a_coord[1] - 1, a_coord[2]]:
                        state.t_pos[a_coord[1] - 1, a_coord[2]] = True
                        state.carrying = 0

                # putting right
                if a_coord[0] == 1:

                    # vase right
                    if state.carrying == 1:

                         # vase doesn't break
                        if state.v_mask[a_coord[1], a_coord[2] + 1]:
                            state.v_pos[a_coord[1], a_coord[2] + 1] = True
                            state.carrying = 0

                        # vase breaks
                        elif state.bv_mask[a_coord[1], a_coord[2] + 1]:
                            state.bv_pos[a_coord[1], a_coord[2] + 1] = True
                            state.carrying = 0

                    # tablecloth right
                    if state.carrying == 2 and state.t_mask[a_coord[1], a_coord[2] + 1] :
                        state.t_pos[a_coord[1], a_coord[2] + 1] = True
                        state.carrying = 0

                # putting down
                if a_coord[0] == 2:

                    # vase down
                    if state.carrying == 1:

                         # vase doesn't break
                        if state.v_mask[a_coord[1] + 1, a_coord[2]]:
                            state.v_pos[a_coord[1] + 1, a_coord[2]] = True
                            state.carrying = 0

                        # vase breaks
                        elif state.bv_mask[a_coord[1] + 1, a_coord[2]]:
                            state.bv_pos[a_coord[1] + 1, a_coord[2]] = True
                            state.carrying = 0

                    # tablecloth down
                    if state.carrying == 2 and state.t_mask[a_coord[1] + 1, a_coord[2]] :
                        state.t_pos[a_coord[1] + 1, a_coord[2]] = True
                        state.carrying = 0

                # putting left
                if a_coord[0] == 3:

                    # vase left
                    if state.carrying == 1:

                         # vase doesn't break
                        if state.v_mask[a_coord[1], a_coord[2] - 1]:
                            state.v_pos[a_coord[1], a_coord[2] - 1] = True
                            state.carrying = 0

                        # vase breaks
                        elif state.bv_mask[a_coord[1], a_coord[2] - 1]:
                            state.bv_pos[a_coord[1], a_coord[2] - 1] = True
                            state.carrying = 0

                    # tablecloth left
                    if state.carrying == 2 and state.t_mask[a_coord[1], a_coord[2] - 1] :
                        state.t_pos[a_coord[1], a_coord[2] - 1] = True
                        state.carrying = 0
        return state
