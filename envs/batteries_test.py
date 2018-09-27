import unittest

from batteries import BatteriesState, BatteriesEnv
from utils import Direction


class TestBatteriesSpec(object):
    def __init__(self):
        """Test spec for the Batteries environment.

        G is a goal location, B is a battery, A is the agent, and T is the train.
        -------
        |B G B|
        |  TT |
        |  TTG|
        |     |
        |A    |
        -------
        """
        self.height = 5
        self.width = 5
        self.init_state = BatteriesState((0, 4), (2, 1), 8,
                                         {(0, 0): True, (4, 4): True},
                                         False)
        self.feature_locations = [(2, 0), (4, 2)]
        self.train_transition = {
            (2, 1): (3, 1),
            (3, 1): (3, 2),
            (3, 2): (2, 2),
            (2, 2): (2, 1)
        }


class TestBatteriesEnv(unittest.TestCase):
    def check_trajectory(self, env, trajectory):
        state = env.s
        for action, next_state in trajectory:
            self.assertEqual(env.state_step(action, state), next_state)
            self.assertEqual(env.state_step(action), next_state)
            features, reward, done, info = env.step(action)
            self.assertEqual(env.s, next_state)
            state = next_state

    def test_trajectories(self):
        batteries_env = BatteriesEnv(TestBatteriesSpec(), compute_transitions=False)
        u, d, l, r, s = map(
            Direction.get_number_from_direction,
            [Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST, Direction.STAY])

        def make_state(agent, train, life, battery_vals, carrying_battery):
            battery_present = dict(zip([(0, 0), (4, 4)], battery_vals))
            return BatteriesState(agent, train, life, battery_present, carrying_battery)

        self.check_trajectory(batteries_env, [
            (u, make_state((0, 3), (3, 1), 7, [True, True], False)),
            (u, make_state((0, 2), (3, 2), 6, [True, True], False)),
            (u, make_state((0, 1), (2, 2), 5, [True, True], False)),
            (u, make_state((0, 0), (2, 1), 4, [False, True], True)),
            (r, make_state((1, 0), (3, 1), 3, [False, True], True)),
            (r, make_state((2, 0), (3, 2), 2, [False, True], True)),
            (s, make_state((2, 0), (2, 2), 1, [False, True], True)),
            (s, make_state((2, 0), (2, 1), 0, [False, True], True)),
            (d, make_state((2, 1), (3, 1), 9, [False, True], False)),
            (u, make_state((2, 0), (3, 2), 8, [False, True], False)),
        ])


if __name__ == '__main__':
    unittest.main()
