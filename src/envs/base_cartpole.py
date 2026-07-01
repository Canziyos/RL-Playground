from dynamics import THETA_LIMIT, angular_accel, linear_accel


class BaseCartPoleSim:
    def __init__(self, cart_mass, pole_mass, length, gravity, force, dt):
        self.c_m = float(cart_mass)
        self.p_m = float(pole_mass)
        self.l = float(length)
        self.g = float(gravity)
        self.F = float(force)
        self.dt = float(dt)

        self.x_threshold = 2.4
        self.theta_threshold = THETA_LIMIT
        self.state = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self.state = [0.0, 0.0, 0.0, 0.0]
        return tuple(self.state)

    @staticmethod
    def _coerce_action(action):
        try:
            return int(action)
        except Exception as exc:
            raise ValueError("Action must be -1 or +1, or 0/1 to map to -1/+1.") from exc

    def _map_action(self, action):
        action = self._coerce_action(action)
        if action == -1:
            return -1
        if action in (0, 1):
            return 1 if action == 1 else -1
        raise ValueError("Action must be -1 or +1, or 0/1 to map to -1/+1.")

    def terminated(self):
        x, _, theta, _ = self.state
        return abs(x) >= self.x_threshold or abs(theta) >= self.theta_threshold

    def step(self, action):
        force = self._map_action(action) * self.F
        x, dx, theta, dtheta = self.state

        lin_ac = linear_accel(self.c_m, self.p_m, self.l, self.g, force, self.state)
        ang_ac = angular_accel(self.c_m, self.p_m, self.l, self.g, force, self.state)

        dx += lin_ac * self.dt
        dtheta += ang_ac * self.dt
        x += dx * self.dt
        theta += dtheta * self.dt

        self.state = [x, dx, theta, dtheta]
        done = self.terminated()
        reward = 0 if done else 1
        return tuple(self.state), reward, done
