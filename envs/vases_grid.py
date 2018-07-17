import numpy as np
import operator as op
from scipy.special import comb
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
    def __init__(self, d_pos, v_pos, bv_pos, agent_pos, t_pos, carrying):
        self.d_pos = d_pos
        self.v_pos = v_pos
        self.bv_pos = bv_pos
        self.agent_pos = agent_pos
        self.t_pos = t_pos
        # Variable determining whether the agent is carrying something:
        # 0 -> nothing, 1 -> vase, 2 -> tablecloth
        self.carrying = carrying


class VasesGrid(object):
    def __init__(self, spec, init_state):
        self.spec = spec
        self.init_state = init_state

        # testing the functions below, they shouldn't be here in the final env
        self.enumerate_states()
        self.step(init_state, 0)

    def enumerate_states(self):
        i = 0
        n_v = self.spec.n_v
        n_t = _v = self.spec.n_t
        # Could be at the same location as the agent, or at any of the ds
        n_v_pos = 1 + np.sum(self.spec.d_mask)
        n_bv_pos = np.sum(self.spec.bv_mask)
        n_agent_pos = np.sum(self.spec.agent_mask)
        n_t_pos = np.sum(self.spec.t_mask)
        self.P = {}

        # Possible agent positions
        for agent_pos in unique_perm(zeros_with_ones(n_agent_pos, 1)):
            agent_mask_pos = np.zeros_like(self.spec.agent_mask.flatten())
            np.put(agent_mask_pos, np.where(self.spec.agent_mask.flatten()), agent_pos)
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
                    for bv_pos in unique_perm(zeros_with_ones(n_bv_pos, n_bv)):
                        bv_mask_pos = np.zeros_like(self.spec.bv_mask.flatten())
                        np.put(bv_mask_pos, np.where(self.spec.bv_mask.flatten()), v_pos)

                        # Possible tablecloth positions; the agent can carry
                        # either a vase or a tablecloth, not both. However,
                        # this is only reflected in the step function since it's
                        # possible for an agent carrying a vase to step on a tile
                        # that contains a dropped tablecloth
                        for t_pos in unique_perm(zeros_with_ones(n_t_pos, n_t)):
                            t_mask_pos = np.zeros_like(self.spec.t_mask.flatten())
                            np.put(t_mask_pos, np.where(self.spec.t_mask.flatten()), t_pos)
                            t_mask_pos = t_mask_pos.reshape(self.spec.t_mask.shape)

                            # print(i); i+=1
                            # TODO enumerate over carrying object / object being
                            # on the floor
                            # TODO add the state to P; how to best store states?


    def step(self, state, action):
        'returns the next state given a state and an action'

        agent_coord = np.where(state.agent_pos)
        # movement
        if action in [0, 1, 2, 3]:
            # move up
            if action==0:
                # no wall above
                if agent_coord[1]!=0:
                    # no desk above
                    if self.spec.d_mask[agent_coord[1]-1, agent_coord[2]]==0:
                        # position on grid
                        print(agent_coord[1]-1, agent_coord[2])

                        agent_coord_new = tuple(map(op.add, agent_coord, (0, -1, 0)))
                        # rotation to up
                        agent_coord_new = (0, agent_coord_new[1], agent_coord_new[2])

                # wall above
                if agent_coord[1]==0:
                    # rotaton to up
                    agent_coord_new = (0, agent_coord[1], agent_coord[2])

            # moving right
            if action==1:
                # no wall to the right
                if agent_coord[2]!=state.agent_pos.shape[2]:
                    # no desk to the right
                    if self.spec.d_mask[agent_coord[1], agent_coord[2]+1]==0:
                        # position on grid
                        agent_coord_new = tuple(map(op.add, agent_coord, (0, 0, 1)))
                        # rotation to the right
                        agent_coord_new = (1, agent_coord_new[1], agent_coord_new[2])

                # wall to the right
                if agent_coord[1]==state.agent_pos.shape[1]:
                    # rotaton to right
                    agent_coord_new = (1, agent_coord[1], agent_coord[2])

            # moving down
            if action==2:
                # no wall below
                if agent_coord[1]!=state.agent_pos.shape[1]:
                    # no desk below
                    if self.spec.d_mask[agent_coord[1]+1, agent_coord[2]]==0:
                        # position on grid
                        agent_coord_new = tuple(map(op.add, agent_coord, (0, 1, 0)))
                        # rotation to down
                        agent_coord_new = (2, agent_coord_new[1], agent_coord_new[2])

                # wall below
                if agent_coord[1]==state.agent_pos.shape[1]:
                    # rotaton to down
                    agent_coord_new = (2, agent_coord[1], agent_coord[2])



            # moving left
            if action==3:
                # no wall to the left
                if agent_coord[2]!=0:
                    # no desk to the left
                    if self.spec.d_mask[agent_coord[1], agent_coord[2]-1]==0:
                        # position on grid
                        agent_coord_new = tuple(map(op.add, agent_coord, (0, 0, -1)))
                        # rotation to left
                        agent_coord_new = (3, agent_coord_new[1], agent_coord_new[2])

                # wall below
                if agent_coord[1]==state.agent_pos.shape[1]:
                    # rotaton to left
                    agent_coord_new = (3, agent_coord[1], agent_coord[2])

            # update agent_pos
            agent_pos_new = np.zeros_like(state.agent_pos)
            agent_pos_new[agent_coord_new] = True
            state.agent_pos = agent_pos_new

            # carrying a vase
            if state.carrying==1:
                # update coord of the vase that was at agent_coord
                state.v_pos[agent_coord[1], agent_coord[2]]=False
                state.v_pos[agent_coord_new[1], agent_coord_new[2]]=True

            if state.carrying==2:
                # Update coord of the tablecloth that was at agent_coord
                state.t_pos[agent_coord[1], agent_coord[2]]=False
                state.t_pos[agent_coord_new[1], agent_coord_new[2]]=True

        # drop object
        if action==5:
            pass
