import numpy as np

class BoxesEnvSpec6x7(object):
    def __init__(self, with_wall):
        '''
        wall_mask shows which surfaces are walls, the rest can be passed through
        goal_mask shows the position of the goal tile
        '''
        self.wall_mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                                   [1, 0, 0, 1, 1, 1, 1],
                                   [1, 0, 0, 0, 0, 0, 1],
                                   [1, 1, 0, 0, 0, 0, 1],
                                   [1, 1, 1, 0, 0, 0, 1],
                                   [1, 1, 1, 1, 1, 1, 1]], dtype='bool')

        self.goal_mask = np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]], dtype='bool')

        self.agent_mask = 1-self.wall_mask
        self.box_mask = 1-self.wall_mask
        self.n_b = 1
        if with_wall:
            self.init_state = BoxesEnvWallState6x7()
        else:
            self.init_state = BoxesEnvNoWallState6x7()


class BoxesEnvWallState6x7(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self):
        self.a_pos = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], dtype='bool')

        self.b_pos = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], dtype='bool')


class BoxesEnvNoWallState6x7(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self):
        self.a_pos = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], dtype='bool')

        self.b_pos = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], dtype='bool')


class BoxesEnvSpec7x9(object):
    def __init__(self, with_wall):
        '''
        wall_mask shows which surfaces are walls, the rest can be passed through
        goal_mask shows the position of the goal tile
        '''
        self.wall_mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                                   [1, 0, 0, 1, 1, 1, 1],
                                   [1, 0, 0, 0, 0, 0, 1],
                                   [1, 1, 0, 0, 0, 0, 1],
                                   [1, 1, 1, 0, 0, 0, 1],
                                   [1, 1, 1, 0, 0, 0, 1],
                                   [1, 1, 1, 0, 0, 0, 1],
                                   [1, 1, 1, 0, 0, 0, 1],
                                   [1, 1, 1, 1, 1, 1, 1]], dtype='bool')

        self.goal_mask = np.array([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]], dtype='bool')

        self.agent_mask = 1-self.wall_mask
        self.box_mask = 1-self.wall_mask
        self.n_b = 3
        if with_wall:
            self.init_state = BoxesEnvWallState7x9()
        else:
            self.init_state = BoxesEnvNoWallState7x9()


class BoxesEnvWallState7x9(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self):
        self.a_pos = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], dtype='bool')

        self.b_pos = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0]], dtype='bool')
 

class BoxesEnvNoWallState7x9(object):
    '''
    state of the environment; describes positions of all objects in the env.
    '''
    def __init__(self):
        self.a_pos = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], dtype='bool')

        self.b_pos = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], dtype='bool')


BOXES_PROBLEMS = {
    "wall:nowall": (BoxesEnvSpec7x9(True), BoxesEnvNoWallState7x9()),
    "nowall:nowall": (BoxesEnvSpec7x9(False), BoxesEnvNoWallState7x9()),
    "wall:nowall_small": (BoxesEnvSpec6x7(True), BoxesEnvNoWallState6x7()),
    "nowall:nowall_small": (BoxesEnvSpec6x7(False), BoxesEnvNoWallState6x7()),
}
