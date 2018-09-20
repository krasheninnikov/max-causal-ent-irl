import numpy as np
from envs.room import RoomState

class RoomSpec(object):
    def __init__(self, height, width, init_state, carpet_locations, feature_locations):
        """See RoomEnv.__init__ in room.py for details."""
        self.height = height
        self.width = width
        self.init_state = init_state
        self.carpet_locations = carpet_locations
        self.feature_locations = feature_locations



# In the diagrams below, G is a goal location, V is a vase, C is a carpet, A is
# the agent.

ROOM_PROBLEMS = {
    # -------
    # |  G  |
    # |GCVC |
    # |  A  |
    # -------
    'simple': (RoomSpec(3, 5,
                        RoomState((2, 2), {(2, 1): True}),
                        [(1, 1), (3, 1)],
                        [(0, 1), (2, 0)]),
               RoomState((2, 0), {(2, 1): True}))
}
