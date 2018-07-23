import numpy as np

class VasesEnvSpec2x3(object):
    def __init__(self):

        self.d_mask = np.array([[1, 0, 1],
                                [0, 0, 0]], dtype='bool')
        self.v_mask = np.array([[1, 0, 1],
                                [0, 0, 0]], dtype='bool')
        self.bv_mask = np.array([[0, 1, 0],
                                 [1, 1, 1]], dtype='bool')
        self.agent_mask = np.array([[[0, 1, 0],
                                    [1, 1, 1]],
                                   [[0, 1, 0],
                                    [1, 1, 1]],
                                   [[0, 1, 0],
                                    [1, 1, 1]],
                                   [[0, 1, 0],
                                    [1, 1, 1]]], dtype='bool')
        self.t_mask = np.array([[1, 1, 1],
                                [1, 1, 1]], dtype='bool')

        self.n_v = 1
        self.n_t = 1

        self.table_mask = np.array([[0, 0, 1],
                                    [0, 0, 0]], dtype='bool')

class VasesEnvState2x3(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self):
        self.d_pos = np.array([[1, 0, 1],
                                [0, 0, 0]], dtype='bool')
        self.v_pos = np.array([[1, 0, 0],
                                [0, 0, 0]], dtype='bool')
        self.bv_pos = np.array([[0, 0, 0],
                                [0, 0, 0]], dtype='bool')
        self.a_pos = np.array([[[0, 1, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0]]], dtype='bool')

        self.t_pos = np.array([[1, 0, 0],
                                [0, 0, 0]], dtype='bool')
        # Variable determining whether the agent is carrying something:
        # [0, 0] -> nothing, [1, 0] -> vase, [0, 1] -> tablecloth
        self.carrying = np.array([0, 0], dtype='bool')


class VasesEnvSpec3x3(object):
    def __init__(self):

        self.d_mask = np.array([[1, 0, 1],
                              [0, 0, 0],
                              [1, 0, 1]], dtype='bool')
        self.table_mask = np.array([[0, 0, 1],
                                      [0, 0, 0],
                                      [0, 0, 0]], dtype='bool')
        self.v_mask = np.array([[1, 0, 1],
                              [0, 0, 0],
                              [1, 0, 1]], dtype='bool')
        self.bv_mask = np.array([[0, 1, 0],
                                 [1, 1, 1]], dtype='bool')
        self.agent_mask = np.array([[[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]],
                                   [[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]],
                                   [[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]],
                                   [[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]]], dtype='bool')

        self.t_mask = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]], dtype='bool')

        self.n_v = 2
        self.n_t = 1


class VasesEnvState3x3(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self):
        self.d_pos = np.array([[1, 0, 1],
                              [0, 0, 0],
                              [1, 0, 1]], dtype='bool')
        self.v_pos = np.array([[1, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0]], dtype='bool')

        self.bv_pos = np.array([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]], dtype='bool')
        self.a_pos = np.array([[[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],
                     [[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]],
                     [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],
                     [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]], dtype='bool')

        self.t_pos = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype='bool')
        # Variable determining whether the agent is carrying something:
        # [0, 0] -> nothing, [1, 0] -> vase, [0, 1] -> tablecloth
        self.carrying = np.array([0, 0], dtype='bool')
