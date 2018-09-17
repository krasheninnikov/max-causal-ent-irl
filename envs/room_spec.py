import numpy as np
from envs.room import RoomState

class RoomSpec(object):
    def __init__(self):
        """See RoomEnv.__init__ in room.py for details.

        Implements the following initial state.
        G is a goal location, V is a vase, C is a carpet, A is the agent.
        -------
        |G G G|
        | CVC |
        |  A  |
        -------
        """
        self.height = 3
        self.width = 5
        self.init_state = RoomState((2, 2), {(2, 1): True})
        self.carpet_locations = [(1, 1), (3, 1)]
        self.feature_locations = [(0, 0), (2, 0), (4, 0)]
