import numpy as np
from envs.apples import ApplesState
from envs.utils import Direction

class ApplesSpec(object):
    def __init__(self, height, width, init_state, apple_regen_probability,
                 bucket_capacity, include_location_features):
        """See ApplesEnv.__init__ in apples.py for details."""
        self.height = height
        self.width = width
        self.init_state = init_state
        self.apple_regen_probability = apple_regen_probability
        self.bucket_capacity = bucket_capacity
        self.include_location_features = include_location_features


# In the diagrams below, T is a tree, B is a bucket, C is a carpet, A is the
# agent. Each tuple is of the form (spec, current state, task R, true R).

APPLES_PROBLEMS = {
    # -----
    # |T T|
    # |   |
    # |AB |
    # -----
    # After 10 timesteps, it looks like this:
    # -----
    # |T T|
    # | A |
    # | B |
    # -----
    # Where the agent has picked both trees once and put the fruit in the
    # basket.
    'default': (
        ApplesSpec(3, 3,
                   ApplesState(agent_pos=(0, 0, 2),
                               tree_states={(0, 0): True, (2, 0): True},
                               bucket_states={(1, 2): 0},
                               carrying_apple=False),
                   apple_regen_probability = 0.1,
                   bucket_capacity=10,
                   include_location_features=True),
        ApplesState(agent_pos=(Direction.get_number_from_direction(Direction.SOUTH),
                               1, 1),
                    tree_states={(0, 0): True, (2, 0): False},
                    bucket_states={(1, 2): 2},
                    carrying_apple=False),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    )
}
