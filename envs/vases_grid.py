import numpy as np
from scipy.special import comb
from utils import unique_perm, zeros_with_ones

class VasesEnvSpec(object):
    def __init__(self, n_v, n_t, d_mask, v_mask, bv_mask, agent_mask, t_mask):
        self.n_v = n_v
        self.n_t = n_t
        self.d_mask = d_mask
        self.v_mask = v_mask
        self.bv_mask = bv_mask
        self.agent_mask = agent_mask
        self.t_mask = t_mask


class VaseEnvState(object):
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
        self.enumerate_states()


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

                            print(i); i+=1
                            # TODO add the state to P; how to best store states?


    def step(self, state, action):
        'returns the next state s_prime given a state and an action'
        # move up
        if action==0:
            pass
