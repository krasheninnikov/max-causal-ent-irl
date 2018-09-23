import unittest

from apples import ApplesState, ApplesEnv
from utils import Direction


class TestApplesSpec(object):
    def __init__(self):
        """Test spec for the Apples environment.

        T is a tree, B is a bucket, C is a carpet, A is the agent.
        -----
        |T T|
        |   |
        |A B|
        -----
        """
        self.height = 3
        self.width = 5
        self.init_state = ApplesState(
            agent_pos=(0, 0, 2),
            tree_states={(0, 0): True, (2, 0): True},
            bucket_states={(2, 2): 0},
            carrying_apple=False)
        # Use a power of 2, to avoid rounding issues
        self.apple_regen_probability = 1.0 / 4
        self.bucket_capacity = 10


class TestApplesEnv(unittest.TestCase):
    def check_trajectory(self, env, trajectory):
        state = env.s
        for action, prob, next_state in trajectory:
            actual_next_states = env.get_next_states(state, action)
            self.assertEqual(sum([p for p, _, _ in actual_next_states]), 1.0)
            self.assertIn((prob, next_state, 0), actual_next_states)
            state = next_state

    def test_trajectories(self):
        u, d, l, r, s = map(
            Direction.get_number_from_direction,
            [Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST, Direction.STAY])
        i = 5  # interact action

        def make_state(agent_pos, tree1, tree2, bucket, carrying_apple):
            tree_states = { (0, 0): tree1, (2, 0): tree2 }
            bucket_state = { (2, 2): bucket }
            return ApplesState(agent_pos, tree_states, bucket_state, carrying_apple)

        apples_env = ApplesEnv(TestApplesSpec(), compute_transitions=False)
        self.check_trajectory(apples_env, [
            (u, 1.0,    make_state((u, 0, 1), True, True, 0, False)),
            (i, 1.0,    make_state((u, 0, 1), False, True, 0, True)),
            (r, 3.0/4,  make_state((r, 1, 1), False, True, 0, True)),
            (r, 3.0/4,  make_state((r, 2, 1), False, True, 0, True)),
            (d, 3.0/4,  make_state((d, 2, 1), False, True, 0, True)),
            (i, 3.0/4,  make_state((d, 2, 1), False, True, 1, False)),
            (u, 3.0/4,  make_state((u, 2, 1), False, True, 1, False)),
            (i, 3.0/4,  make_state((u, 2, 1), False, False, 1, True)),
            (d, 9.0/16, make_state((d, 2, 1), False, False, 1, True)),
            (i, 3.0/16, make_state((d, 2, 1), True, False, 2, False)),
            (s, 1.0/4,  make_state((d, 2, 1), True, True, 2, False)),
        ])

if __name__ == '__main__':
    unittest.main()
