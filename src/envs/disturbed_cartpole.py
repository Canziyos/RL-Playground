import math
import random

from dynamics import THETA_LIMIT, angular_accel, linear_accel


class DisturbedCartPoleSim:
    """CartPole with optional disturbances. "Drunk CartPole" is the project nickname."""

    def __init__(self, cart_mass, pole_mass, length, gravity, force, dt, disturbances=None, rng=None):
        self.base_c_m = float(cart_mass)
        self.base_p_m = float(pole_mass)
        self.base_l = float(length)
        self.base_g = float(gravity)
        self.base_F = float(force)
        self.dt = float(dt)

        disturbances = disturbances or {}
        self.wind_bias = float(disturbances.get("wind_bias", 0.0))
        self.wind_gust_std = float(disturbances.get("wind_gust_std", 0.0))
        self.shake_amp = float(disturbances.get("shake_amp", 0.0))
        self.shake_freq_hz = float(disturbances.get("shake_freq_hz", 0.0))
        self.impulse_prob = float(disturbances.get("impulse_prob", 0.0))
        self.impulse_mag = float(disturbances.get("impulse_mag", 0.0))
        self.obs_noise = tuple(disturbances.get("obs_noise", (0.0, 0.0, 0.0, 0.0)))
        self.param_jitter = float(disturbances.get("param_jitter", 0.0))

        self.x_limit = 2.4
        self.rng = rng if rng is not None else random.Random()
        self.t = 0.0
        self.c_m = self.base_c_m
        self.p_m = self.base_p_m
        self.l = self.base_l
        self.g = self.base_g
        self.F = self.base_F
        self.state = [0.0, 0.0, 0.0, 0.0]

    def _apply_param_jitter(self):
        if self.param_jitter <= 0.0:
            self.c_m = self.base_c_m
            self.p_m = self.base_p_m
            self.l = self.base_l
            self.g = self.base_g
            self.F = self.base_F
            return

        eps = self.param_jitter

        def jitter(value):
            return value * (1.0 + self.rng.uniform(-eps, eps))

        self.c_m = jitter(self.base_c_m)
        self.p_m = jitter(self.base_p_m)
        self.l = jitter(self.base_l)
        self.g = jitter(self.base_g)
        self.F = jitter(self.base_F)

    def reset(self, drunk=False):
        self._apply_param_jitter()
        self.t = 0.0
        x = self.rng.uniform(-0.05, 0.05)
        dx = 0.0
        theta = self.rng.uniform(-0.05, 0.05)
        dtheta = 0.0
        if drunk:
            dx += self.rng.uniform(-0.5, 0.5)
            dtheta += self.rng.uniform(-0.5, 0.5)
        self.state = [x, dx, theta, dtheta]
        return tuple(self._observe())

    def _disturbance_force(self):
        force = 0.0
        if self.wind_bias != 0.0:
            force += self.wind_bias
        if self.wind_gust_std > 0.0:
            force += self.rng.gauss(0.0, self.wind_gust_std)
        if self.shake_amp != 0.0 and self.shake_freq_hz != 0.0:
            force += self.shake_amp * math.sin(2.0 * math.pi * self.shake_freq_hz * self.t)
        if self.impulse_prob > 0.0 and self.rng.random() < self.impulse_prob:
            sign = -1 if self.rng.random() < 0.5 else 1
            force += sign * self.impulse_mag
        return force

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

    def step(self, action):
        action = self._map_action(action)
        x, dx, theta, dtheta = self.state
        force_total = self.F * action + self._disturbance_force()

        ang_ac = angular_accel(self.c_m, self.p_m, self.l, self.g, force_total, self.state)
        lin_ac = linear_accel(self.c_m, self.p_m, self.l, self.g, force_total, self.state)

        dx += lin_ac * self.dt
        dtheta += ang_ac * self.dt
        x += dx * self.dt
        theta += dtheta * self.dt
        self.state = [x, dx, theta, dtheta]
        self.t += self.dt

        done = self.terminated()
        reward = 0 if done else 1
        return tuple(self._observe()), reward, done

    def _observe(self):
        if not self.obs_noise or all(value == 0.0 for value in self.obs_noise):
            return self.state

        x, dx, theta, dtheta = self.state
        nx, ndx, nt, ndt = self.obs_noise
        if nx != 0.0:
            x += self.rng.gauss(0.0, nx)
        if ndx != 0.0:
            dx += self.rng.gauss(0.0, ndx)
        if nt != 0.0:
            theta += self.rng.gauss(0.0, nt)
        if ndt != 0.0:
            dtheta += self.rng.gauss(0.0, ndt)
        return [x, dx, theta, dtheta]

    def terminated(self):
        x, _, theta, _ = self.state
        return abs(theta) > THETA_LIMIT or abs(x) > self.x_limit
