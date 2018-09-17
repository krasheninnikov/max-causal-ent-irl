import numpy as np

from envs.vases_grid import VasesEnvState


class VasesEnvSpec2x3V2D3(object):
    def __init__(self):
        '''
        d_mask corresponds to all surfaces, desks and tables.
        table_mask shows which surfaces in d_mask are tables; the rest are desks.
        '''
        self.d_mask = np.array([[1, 1, 1],
                                [0, 0, 0]], dtype='bool')
        self.bv_mask = np.array([[0, 0, 0],
                                 [1, 1, 1]], dtype='bool')
        self.agent_mask = np.array([[[0, 0, 0],
                                    [1, 1, 1]],
                                   [[0, 0, 0],
                                    [1, 1, 1]],
                                   [[0, 0, 0],
                                    [1, 1, 1]],
                                   [[0, 0, 0],
                                    [1, 1, 1]]], dtype='bool')
        self.t_mask = np.array([[1, 1, 1],
                                [1, 1, 1]], dtype='bool')

        self.n_v = 2
        self.n_t = 1

        self.table_mask = np.array([[0, 0, 1],
                                    [0, 0, 0]], dtype='bool')
        self.init_state = VasesEnvState2x3V2D3


VasesEnvState2x3V2D3 = VasesEnvState(
    np.array([[1, 0, 1],
              [0, 0, 0]], dtype='bool'),
    np.array([[0, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([[[0, 0, 0],
               [1, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]]], dtype='bool'),
    np.array([[1, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([0, 0], dtype='bool')
)


class VasesEnvSpec2x3(object):
    def __init__(self):
        '''
        d_mask corresponds to all surfaces, desks and tables.
        table_mask shows which surfaces in d_mask are tables; the rest are desks.
        '''
        self.d_mask = np.array([[1, 0, 1],
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
        self.init_state = VasesEnvState2x3


VasesEnvState2x3 = VasesEnvState(
    np.array([[1, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([[0, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([[[0, 1, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]]], dtype='bool'),
    np.array([[1, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([0, 0], dtype='bool')
)


class VasesEnvSpec2x3Broken(object):
    def __init__(self):
        '''
        d_mask corresponds to all surfaces, desks and tables.
        table_mask shows which surfaces in d_mask are tables; the rest are desks.
        '''

        self.d_mask = np.array([[1, 0, 1],
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

        self.n_v = 3
        self.n_t = 1

        self.table_mask = np.array([[0, 0, 1],
                                    [0, 0, 0]], dtype='bool')
        self.init_state = VasesEnvState2x3Broken


VasesEnvState2x3Broken = VasesEnvState(
    np.array([[1, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([[0, 0, 0],
              [1, 0, 1]], dtype='bool'),
    np.array([[[0, 1, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 0, 0]]], dtype='bool'),
    np.array([[1, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([0, 0], dtype='bool')
)


class VasesEnvSpec2x4Broken(object):
    def __init__(self):
        '''
        d_mask corresponds to all surfaces, desks and tables.
        table_mask shows which surfaces in d_mask are tables; the rest are desks.
        '''

        self.d_mask = np.array([[1, 0, 1, 0],
                                [0, 0, 0, 0]], dtype='bool')
        self.bv_mask = np.array([[0, 1, 0, 1],
                                 [1, 1, 1, 1]], dtype='bool')
        self.agent_mask = np.array([[[0, 1, 0, 1],
                                    [1, 1, 1, 1]],
                                   [[0, 1, 0, 1],
                                    [1, 1, 1, 1]],
                                   [[0, 1, 0, 1],
                                    [1, 1, 1, 1]],
                                   [[0, 1, 0, 1],
                                    [1, 1, 1, 1]]], dtype='bool')
        self.t_mask = np.array([[1, 1, 1, 1],
                                [1, 1, 1, 1]], dtype='bool')

        self.n_v = 3
        self.n_t = 1

        self.table_mask = np.array([[0, 0, 1, 0],
                                    [0, 0, 0, 0]], dtype='bool')
        self.init_state = VasesEnvState2x4Broken


VasesEnvState2x4Broken = VasesEnvState(
    np.array([[1, 0, 0, 0],
              [0, 0, 0, 0]], dtype='bool'),
    np.array([[0, 0, 0, 0],
              [1, 0, 1, 0]], dtype='bool'),
    np.array([[[0, 1, 0, 0],
               [0, 0, 0, 0]],
              [[0, 0, 0, 0],
               [0, 0, 0, 0]],
              [[0, 0, 0, 0],
               [0, 0, 0, 0]],
              [[0, 0, 0, 0],
               [0, 0, 0, 0]]], dtype='bool'),
    np.array([[1, 0, 0, 0],
              [0, 0, 0, 0]], dtype='bool'),
    np.array([0, 0], dtype='bool')
)

        
class VasesEnvSpec3x3(object):
    def __init__(self):
        '''
        d_mask corresponds to all surfaces, desks and tables.
        table_mask shows which surfaces in d_mask are tables; the rest are desks.
        '''
        self.d_mask = np.array([[1, 0, 1],
                              [0, 0, 0],
                              [1, 0, 1]], dtype='bool')
        self.table_mask = np.array([[0, 0, 1],
                                      [0, 0, 0],
                                      [0, 0, 0]], dtype='bool')
        self.bv_mask = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]], dtype='bool')
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
        self.init_state = VasesEnvState3x3


VasesEnvState3x3 = VasesEnvState(
    np.array([[1, 0, 0],
              [0, 0, 0],
              [1, 0, 0]], dtype='bool'),
    np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([[[0, 0, 0],
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
               [0, 0, 0]]], dtype='bool'),
    np.array([[1, 0, 0],
              [0, 0, 0],
              [0, 0, 0]], dtype='bool'),
    np.array([0, 0], dtype='bool')
)


VASES_PROBLEMS = {
    # TODO: This is wrong, it sets the current state equal to the initial state
    'default': (VasesEnvSpec2x3V2D3(), VasesEnvState2x3V2D3)
}
