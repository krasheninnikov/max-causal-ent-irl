import numpy as np
from envs.room import RoomState

class RoomSpec(object):
    def __init__(self, height, width, init_state, carpet_locations, feature_locations):
        """See RoomEnv.__init__ in room.py for details.

        Implements the following initial state.
        G is a goal location, V is a vase, C is a carpet, A is the agent.
        -------
        |G G G|
        | CVC |
        |  A  |
        -------
        """
        self.height = height
        self.width = width
        self.init_state = init_state
        self.carpet_locations = carpet_locations
        self.feature_locations = feature_locations


ROOM_PROBLEMS = {
    'simple': (RoomSpec(3, 5,
                        RoomState((2, 2), {(2, 1): True}),
                        [(1, 1), (3, 1)],
                        [(0, 0), (2, 0), (4, 0)]),
               RoomState((2, 2), {(2, 1): True}))
}
