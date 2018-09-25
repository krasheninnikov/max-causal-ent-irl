import unittest

from train import TrainState, TrainEnv
from utils import Direction


class TestTrainSpec(object):
    def __init__(self):
        """Test spec for the Train environment.

        G is a goal location, V is a vase, C is a carpet, A is the agent.
        -------
        |  G  |
        |  TT |
        |CVTTG|
        |  V  |
        |A    |
        -------
        """
        self.height = 5
        self.width = 5
        self.init_state = TrainState((0, 4), {(1, 2): True, (2, 3): True}, (2, 1), True)
        self.carpet_locations = [(0, 2)]
        self.feature_locations = [(2, 0), (4, 2)],
        self.train_transition = {
            (2, 1): (3, 1),
            (3, 1): (3, 2),
            (3, 2): (2, 2),
            (2, 2): (2, 1)
        }


class TestTrainEnv(unittest.TestCase):
    def check_trajectory(self, env, trajectory):
        state = env.s
        for action, next_state in trajectory:
            self.assertEqual(env.state_step(action, state), next_state)
            self.assertEqual(env.state_step(action), next_state)
            features, reward, done, info = env.step(action)
            self.assertEqual(env.s, next_state)
            state = next_state

    def test_trajectories(self):
        train_env = TrainEnv(TestTrainSpec(), compute_transitions=False)
        u, d, l, r, s = map(
            Direction.get_number_from_direction,
            [Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST, Direction.STAY])

        self.check_trajectory(train_env, [
            (u, TrainState((0, 3), {(1, 2): True, (2, 3): True}, (3, 1), True)),
            (u, TrainState((0, 2), {(1, 2): True, (2, 3): True}, (3, 2), True)),
            (u, TrainState((0, 1), {(1, 2): True, (2, 3): True}, (2, 2), True)),
            (r, TrainState((1, 1), {(1, 2): True, (2, 3): True}, (2, 1), True)),
            (u, TrainState((1, 0), {(1, 2): True, (2, 3): True}, (3, 1), True)),
            (r, TrainState((2, 0), {(1, 2): True, (2, 3): True}, (3, 2), True)),
            (s, TrainState((2, 0), {(1, 2): True, (2, 3): True}, (2, 2), True)),
            (s, TrainState((2, 0), {(1, 2): True, (2, 3): True}, (2, 1), True)),
        ])

        train_env.reset()
        self.check_trajectory(train_env, [
            (u, TrainState((0, 3), {(1, 2): True, (2, 3): True}, (3, 1), True)),
            (r, TrainState((1, 3), {(1, 2): True, (2, 3): True}, (3, 2), True)),
            (r, TrainState((2, 3), {(1, 2): True, (2, 3): False}, (2, 2), True)),
        ])

        train_env.reset()
        self.check_trajectory(train_env, [
            (r, TrainState((1, 4), {(1, 2): True, (2, 3): True}, (3, 1), True)),
            (r, TrainState((2, 4), {(1, 2): True, (2, 3): True}, (3, 2), True)),
            (r, TrainState((3, 4), {(1, 2): True, (2, 3): True}, (2, 2), True)),
            (u, TrainState((3, 3), {(1, 2): True, (2, 3): True}, (2, 1), True)),
            (u, TrainState((3, 2), {(1, 2): True, (2, 3): True}, (3, 1), True)),
            (s, TrainState((3, 2), {(1, 2): True, (2, 3): True}, (3, 2), False)),
            (s, TrainState((3, 2), {(1, 2): True, (2, 3): True}, (3, 2), False)),
            (u, TrainState((3, 1), {(1, 2): True, (2, 3): True}, (3, 2), False)),
            (l, TrainState((2, 1), {(1, 2): True, (2, 3): True}, (3, 2), False)),
        ])

if __name__ == '__main__':
    unittest.main()
