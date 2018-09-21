import numpy as np
from collections import defaultdict
from copy import deepcopy

class Env(object):
    def __init__(self):
        raise ValueError('Cannot instantiate abstract class Env')

    def is_deterministic(self):
        return False

    def make_transition_matrices(self, states_iter, actions_iter):
        """
        states_iter: ITERATOR of states (i.e. can only be used once)
        actions_iter: ITERATOR of actions (i.e. can only be used once)
        """
        P = {}
        actions = list(actions_iter)
        for state in states_iter:
            state_id = self.get_num_from_state(state)
            P[state_id] = {}
            for action in actions:
                next_s = self.get_next_states(state, action)
                next_s = [(p, self.get_num_from_state(s), r) for p, s, r in next_s]
                P[state_id][action] = next_s

        self.P = P


    def make_f_matrix(self, nS, num_features):
        self.f_matrix = np.zeros((nS, num_features))
        for state_id in self.P.keys():
            state = self.get_state_from_num(state_id)
            self.f_matrix[state_id, :] = self.s_to_f(state)


    def reset(self):
        self.timestep = 0
        self.s = deepcopy(self.init_state)

        obs = self.s_to_f(self.s)
        return np.array(obs, dtype='float32').flatten() #, obs.T @ self.r_vec, False, defaultdict(lambda : '')

    def state_step(self, action, state=None):
        if state == None: state = self.s
        next_states = self.get_next_states(state, action)
        probabilities = [p for p, _ in next_states]
        idx = np.random.choice(np.arange(len(next_states)), p=probabilities)
        return next_states[idx][1]

    def step(self, action, r_vec=None):
        '''
        given an action, takes a step from self.s, updates self.s and returns:
        - the observation (features of the next state)
        - the associated reward
        - done, the indicator of completed episode
        - info
        '''
        self.s = self.state_step(action)
        self.timestep+=1

        obs = self.s_to_f(self.s)
        reward = 0 if r_vec is None else np.array(obs.T @ r_vec)
        done = False
        info = defaultdict(lambda : '')
        return np.array(obs, dtype='float32'), reward, np.array(done, dtype='bool'), info


class DeterministicEnv(Env):
    def __init__(self):
        raise ValueError('Cannot instantiate abstract class DeterministicEnv')

    def is_deterministic(self):
        return True

    def make_transition_matrices(self, states_iter, actions_iter, nS, nA):
        """
        states_iter: ITERATOR of states (i.e. can only be used once)
        actions_iter: ITERATOR of actions (i.e. can only be used once)
        nS: Number of states
        nA: Number of actions
        """
        Env.make_transition_matrices(self, states_iter, actions_iter)
        self._make_stochastic_transition_matrix(nS, nA)
        self._make_deterministic_transition_matrix(nS, nA)
        self._make_deterministic_transition_transpose_matrix(nS, nA)


    def get_next_states(self, state, action):
        return [(1, self.get_next_state(state, action), 0)]

    def state_step(self, action, state=None):
        if state == None: state = self.s
        return self.get_next_state(state, action)

    def _make_stochastic_transition_matrix(self, nS, nA):
        '''Create self.T, a matrix with index S,A,S' -> P(S'|S,A)      '''
        self.T = np.zeros([nS, nA, nS])
        for s in range(nS):
            for a in range(nA):
                self.T[s, a, self.P[s][a][0][1]] = 1

    def _make_deterministic_transition_matrix(self, nS, nA):
        '''Create self.deterministic_T, a matrix with index S,A -> S'   '''
        self.deterministic_T = np.zeros((nS, nA), dtype='int32')
        for s in range(nS):
            for a in range(nA):
                self.deterministic_T[s,a]=self.P[s][a][0][1]

    def _make_deterministic_transition_transpose_matrix(self, nS, nA):
        '''Create self.deterministic_transpose, a matrix with index S,A -> S', with the inverse dynamics '''
        self.deterministic_transpose = np.zeros((nS, nA), dtype='int32')
        for s in range(nS):
            for a in range(nA):
                self.deterministic_transpose[self.P[s][a][0][1],a]=s
