import random


class QAgent:
    def __init__(self, n_actions=2, alpha=None, gamma=None, epsilon=None):
        self.n_act = int(n_actions)
        if self.n_act < 1:
            raise ValueError("n_actions must be >= 1.")

        self.alpha = alpha
        if self.alpha is None or not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError("learning_rate must be in ]0, 1].")

        self.gamma = gamma
        if self.gamma is None or not (0.0 < float(self.gamma) <= 1.0):
            raise ValueError("Discount Factor must be in ]0, 1].")

        self.epsilon = epsilon
        if self.epsilon is None or not (0.0 <= float(self.epsilon) <= 1.0):
            raise ValueError("exploration_prob must be in [0, 1].")

        self.alpha = float(self.alpha)
        self.gamma = float(self.gamma)
        self.epsilon = float(self.epsilon)

        self.rng = random.Random(0)
        self.Q = {}

    @staticmethod
    def _check_state(state):
        if len(state) != 4:
            raise ValueError("state must be a 4-tuple: (theta_i, dtheta_i, x_i, dx_i).")
        return tuple(int(i) for i in state)

    def _q(self, state, action):
        return self.Q.get(state + (int(action),), 0.0)

    def _q_values(self, state):
        return [self._q(state, action) for action in range(self.n_act)]

    def select_action(self, state):
        state = self._check_state(state)

        if self.rng.random() < self.epsilon:
            return int(self.rng.randrange(self.n_act))

        q_vals = self._q_values(state)
        max_q = max(q_vals)
        best = [action for action, q in enumerate(q_vals) if q == max_q]
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, done=False):
        state = self._check_state(state)
        next_state = self._check_state(next_state)
        action = int(action)

        q_sa = self._q(state, action)
        if done:
            td_target = float(reward)
        else:
            td_target = float(reward) + self.gamma * max(self._q_values(next_state))

        delta = td_target - q_sa
        new_val = q_sa + self.alpha * delta
        self.Q[state + (action,)] = new_val
        return new_val, delta
