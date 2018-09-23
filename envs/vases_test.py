import numpy as np
import unittest

from vases_grid import VasesEnvState, VasesGrid
from vases_spec import VasesEnvSpec3x3, VasesEnvInitialState3x3, VasesEnvFinalState3x3
from utils import Direction


class TestVasesGrid(unittest.TestCase):
    def check_trajectory(self, env, trajectory, reset=True):
        state = env.s
        for action, next_state in trajectory:
            self.assertEqual(env.state_step(action, state), next_state)
            self.assertEqual(env.state_step(action), next_state)
            features, reward, done, info = env.step(action)
            self.assertEqual(env.s, next_state)
            state = next_state

    def test_trajectories(self):
        vases = VasesGrid(VasesEnvSpec3x3(), compute_transitions=False)
        u, r, d, l, s, i = range(6)

        def make_state(agent_pos, vase1, vase2, tablecloth1, tablecloth2):
            v_pos = np.zeros((3, 3))
            bv_pos = np.zeros((3, 3))
            a_pos = np.zeros((4, 3, 3))
            t_pos = np.zeros((3, 3))
            carrying = np.array([0, 0])

            def handle_thing(thing, x, y, status):
                if thing == 'vase' and status == 'intact':
                    v_pos[y,x] = 1
                elif thing == 'vase' and status == 'broken':
                    bv_pos[y,x] = 1
                elif thing == 'vase' and status == 'carried':
                    carrying[0] = 1
                elif thing == 'tablecloth' and status == 'normal':
                    t_pos[y,x] = 1
                elif thing == 'tablecloth' and status == 'carried':
                    carrying[1] = 1
                else:
                    raise ValueError('Invalid')

            handle_thing('vase', *vase1)
            handle_thing('vase', *vase2)
            handle_thing('tablecloth', *tablecloth1)
            handle_thing('tablecloth', *tablecloth2)
            orientation, x, y = agent_pos
            a_pos[orientation,y,x] = 1

            return VasesEnvState(v_pos, bv_pos, a_pos, t_pos, carrying)

        # |VAV|
        # |T  |
        # | TX|
        # Changes to
        # |VAT|
        # |   |
        # |  V|
        self.check_trajectory(vases, [
            (d, make_state((d, 1, 1), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (1, 2, 'normal'))),
            (i, make_state((d, 1, 1), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (1, 1, 'carried'))),
            (d, make_state((d, 1, 2), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (1, 2, 'carried'))),
            (r, make_state((r, 1, 2), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (1, 2, 'carried'))),
            (i, make_state((r, 1, 2), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (u, make_state((u, 1, 1), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (s, make_state((u, 1, 1), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (r, make_state((r, 2, 1), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (u, make_state((u, 2, 1), (0, 0, 'intact'), (2, 0, 'intact'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (i, make_state((u, 2, 1), (0, 0, 'intact'), (2, 1, 'carried'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (d, make_state((d, 2, 1), (0, 0, 'intact'), (2, 1, 'carried'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (i, make_state((d, 2, 1), (0, 0, 'intact'), (2, 2, 'intact'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (l, make_state((l, 1, 1), (0, 0, 'intact'), (2, 2, 'intact'), (0, 1, 'normal'), (2, 2, 'normal'))),
            (i, make_state((l, 1, 1), (0, 0, 'intact'), (2, 2, 'intact'), (1, 1, 'carried'), (2, 2, 'normal'))),
            (u, make_state((u, 1, 0), (0, 0, 'intact'), (2, 2, 'intact'), (1, 0, 'carried'), (2, 2, 'normal'))),
            (r, make_state((r, 1, 0), (0, 0, 'intact'), (2, 2, 'intact'), (1, 0, 'carried'), (2, 2, 'normal'))),
            (i, VasesEnvFinalState3x3)
        ])

if __name__ == '__main__':
    unittest.main()
