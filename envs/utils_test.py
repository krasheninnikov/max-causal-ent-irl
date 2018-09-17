import unittest

from utils import Direction, all_boolean_assignments

class TestUtils(unittest.TestCase):
    def test_direction_number_conversion(self):
        all_directions = Direction.ALL_DIRECTIONS
        all_numbers = []

        for direction in Direction.ALL_DIRECTIONS:
            number = Direction.get_number_from_direction(direction)
            direction_again = Direction.get_direction_from_number(number)
            self.assertEqual(direction, direction_again)
            all_numbers.append(number)

        # Check that all directions are distinct
        num_directions = len(all_directions)
        self.assertEqual(len(set(all_directions)), num_directions)
        # Check that the numbers are 0, 1, ... num_directions - 1
        self.assertEqual(set(all_numbers), set(range(num_directions)))

    def test_boolean_assignments(self):
        self.assertEqual(list(all_boolean_assignments(1)), [[True], [False]])

        def valid_asgn(asgn, n):
            return type(asgn) == list and len(asgn) == n and \
                all([x in [True, False] for x in asgn])

        asgns = list(all_boolean_assignments(5))
        self.assertEqual(len(asgns), 32)
        self.assertEqual(len(set((tuple(x) for x in asgns))), 32)
        self.assertTrue(all(valid_asgn(asgn, 5) for asgn in asgns))


if __name__ == '__main__':
    unittest.main()
