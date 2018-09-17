import unittest

from room import RoomState, RoomEnv
from utils import Direction


class TestRoomSpec(object):
    def __init__(self):
        """Implements an initial state for Room.

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


class TestRoomEnv(unittest.TestCase):
    def setUp(self):
        self.room = RoomEnv(TestRoomSpec(), compute_transitions=False)
        u, d, l, r = map(
            Direction.get_number_from_direction,
            [Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST])

        self.trajectory1 = [
            (l, RoomState((1, 2), {(2, 1): True})),
            (u, RoomState((1, 1), {(2, 1): True})),
            (u, RoomState((1, 0), {(2, 1): True})),
            (r, RoomState((2, 0), {(2, 1): True}))
        ]
        self.trajectory2 = [
            (u, RoomState((2, 1), {(2, 1): False})),
            (u, RoomState((2, 0), {(2, 1): False}))
        ]
        self.trajectory3 = [
            (r, RoomState((3, 2), {(2, 1): True})),
            (u, RoomState((3, 1), {(2, 1): True})),
            (l, RoomState((2, 1), {(2, 1): False})),
            (d, RoomState((2, 2), {(2, 1): False}))
        ]

    def check_trajectory(self, env, trajectory, reset=True):
        if reset:
            env.reset()

        state = env.s
        for action, next_state in trajectory:
            self.assertEqual(env.state_step(action, state), next_state)
            self.assertEqual(env.state_step(action), next_state)
            features, reward, done, info = env.step(action)
            self.assertEqual(env.s, next_state)
            state = next_state

    def test_trajectories(self):
        self.check_trajectory(self.room, self.trajectory1, reset=False)
        self.check_trajectory(self.room, self.trajectory2)
        self.check_trajectory(self.room, self.trajectory3)

if __name__ == '__main__':
    unittest.main()
