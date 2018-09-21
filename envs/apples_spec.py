import numpy as np
from envs.apples import ApplesState
from envs.utils import Direction

class ApplesSpec(object):
    def __init__(self, height, width, init_state):
        """See ApplesEnv.__init__ in room.py for details."""
        self.height = height
        self.width = width
        self.init_state = init_state


# In the diagrams below, T is a tree, B is a bucket, C is a carpet, A is the
# agent. Each tuple is of the form (spec, current state, task R, true R).

APPLES_PROBLEMS = {
    # -----
    # |T T|
    # |   |
    # |A B|
    # -----
    # After 10 timesteps, it looks like this:
    # -----
    # |T T|
    # |  A|
    # |  B|
    # -----
    # Where the agent has picked both trees once and put the fruit in the
    # basket.
    'default': (
        ApplesSpec(3, 3,
                   ApplesState(agent_pos=(0, 0, 2),
                               tree_states={(0, 0): 10, (2, 0): 10},
                               bucket_states={(2, 2): 0},
                               carrying_apple=False)),
        ApplesState(agent_pos=(Direction.get_number_from_direction(Direction.SOUTH),
                               2, 1),
                    tree_states={(0, 0): 8, (2, 0): 2},
                    bucket_states={(2, 2): 2},
                    carrying_apple=False),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    )
}
