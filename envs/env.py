import numpy as np

class DeterministicEnv(object):
    def __init__(self):
        raise ValueError('Cannot instantiate abstract class Env')

    def make_transition_matrices(self, states, actions):
        self._compute_transitions(states, actions)
        self._make_stochastic_transition_matrix()
        self._make_deterministic_transition_matrix()
        self._make_deterministic_transition_transpose_matrix()

    def make_f_matrix(self):
         self.f_matrix = np.zeros((self.nS, self.num_features))
         for state_str, state_num_id in self.state_num.items():
             self.f_matrix[state_num_id, :] = self.s_to_f(self.str_to_state(state_str))

    def _compute_transitions(self, states, actions):
        # Take every possible action from each of the possible states. Since the
        # env is deterministic, this is sufficient to get transition probs
        P = {}
        for state in states:
            state_str = self.state_to_str(state)
            state_num_id = self.get_state_num(state_str)
            P[state_num_id] = {}
            for action in range(self.nA):
                statep = self.state_step(action, self.str_to_state(state_str))
                statep_str = self.state_to_str(statep)
                statep_num_id = self.get_state_num(statep_str)
                P[state_num_id][action] = [(1, statep_num_id, 0)]

        self.P = P

    def _make_stochastic_transition_matrix(self):
        '''Create self.T, a matrix with index S,A,S' -> P(S'|S,A)      '''
        self.T = np.zeros([self.nS, self.nA, self.nS])
        for s in range(self.nS):
            for a in range(self.nA):
                self.T[s, a, self.P[s][a][0][1]] = 1

    def _make_deterministic_transition_matrix(self):
        '''Create self.deterministic_T, a matrix with index S,A -> S'   '''
        self.deterministic_T = np.zeros((self.nS, self.nA), dtype='int32')
        for s in range(self.nS):
            for a in range(self.nA):
                self.deterministic_T[s,a]=self.P[s][a][0][1]

    def _make_deterministic_transition_transpose_matrix(self):
        '''Create self.deterministic_transpose, a matrix with index S,A -> S', with the inverse dynamics '''
        self.deterministic_transpose = np.zeros((self.nS, self.nA), dtype='int32')
        for s in range(self.nS):
            for a in range(self.nA):
                self.deterministic_transpose[self.P[s][a][0][1],a]=s
