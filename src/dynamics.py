import math


THETA_LIMIT = math.radians(12.0)


def angular_accel(c_m, p_m, length, gravity, force, state):
    _, _, theta, dtheta = state

    num = gravity * math.sin(theta) + math.cos(theta) * (
        (-force) - p_m * length * (dtheta**2) * math.sin(theta)
    ) / (c_m + p_m)
    den = length * ((4.0 / 3.0) - (p_m * (math.cos(theta) ** 2)) / (c_m + p_m))
    return num / den


def linear_accel(c_m, p_m, length, gravity, force, state):
    _, _, theta, dtheta = state
    ang_ac = angular_accel(c_m, p_m, length, gravity, force, state)
    accel = force + p_m * length * (math.sin(theta) * (dtheta**2) - ang_ac * math.cos(theta))
    return accel / (c_m + p_m)
