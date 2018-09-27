import numpy as np
from envs.train import TrainState

class TrainSpec(object):
    def __init__(self, height, width, init_state, carpet_locations, feature_locations, train_transition):
        """See TrainEnv.__init__ in train.py for details."""
        self.height = height
        self.width = width
        self.init_state = init_state
        self.carpet_locations = carpet_locations
        self.feature_locations = feature_locations
        self.train_transition = train_transition



# In the diagrams below, G is a goal location, V is a vase, C is a carpet, A is
# the agent, and T is the train.
# Each tuple is of the form (spec, current state, task R, true R).

TRAIN_PROBLEMS = {
    # -------
    # |  G C|
    # |  TT |
    # | VTTG|
    # |     |
    # |A    |
    # -------
    'default': (
        TrainSpec(5, 5,
                  TrainState((0, 4), {(1, 2): True}, (2, 1), True),
                  [(4, 0)],
                  [(2, 0), (4, 2)],
                  {
                      (2, 1): (3, 1),
                      (3, 1): (3, 2),
                      (3, 2): (2, 2),
                      (2, 2): (2, 1)
                  }),
        TrainState((2, 0), {(1, 2): True}, (2, 2), True),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
        np.array([-1, 0, -1, 0, 0, 0, 0, 0, 1])
    )
}
