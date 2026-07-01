from bisect import bisect_right

from dynamics import THETA_LIMIT


BASE_DISCRETIZATION = {
    "x": (-1.5, -0.5, 0.5, 1.5),
    "theta": (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12),
    "dx": (-1.5, -0.5, 0.5, 1.5),
    "dtheta": (-4.0, -2.0, -0.7, 0.7, 2.0, 4.0),
}

DISTURBED_DISCRETIZATION = {
    "x": (-1.6, -0.8, -0.3, 0.3, 0.8, 1.6),
    "theta": (-THETA_LIMIT, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, THETA_LIMIT),
    "dx": (-0.2, 0.2),
    "dtheta": (-4.0, -2.0, -0.7, 0.7, 2.0, 4.0),
}


def state_index(value, edges):
    return int(bisect_right(edges, value))


def discretize_state(state, preset=BASE_DISCRETIZATION):
    x, dx, theta, dtheta = state
    return (
        state_index(theta, preset["theta"]),
        state_index(dtheta, preset["dtheta"]),
        state_index(x, preset["x"]),
        state_index(dx, preset["dx"]),
    )


def discretize_base_state(state):
    return discretize_state(state, BASE_DISCRETIZATION)


def discretize_disturbed_state(state):
    return discretize_state(state, DISTURBED_DISCRETIZATION)
