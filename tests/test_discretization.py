from src.discretization import discretize_base_state, discretize_disturbed_state


def test_discretization_returns_four_integer_bins():
    state = (0.0, 0.0, 0.0, 0.0)
    assert discretize_base_state(state) == (4, 3, 2, 2)
    assert all(isinstance(value, int) for value in discretize_disturbed_state(state))
    assert len(discretize_disturbed_state(state)) == 4
