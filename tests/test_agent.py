import pytest

from src.agent import QAgent


def test_agent_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError):
        QAgent(alpha=0.0, gamma=0.9, epsilon=0.1)
    with pytest.raises(ValueError):
        QAgent(alpha=0.1, gamma=1.1, epsilon=0.1)
    with pytest.raises(ValueError):
        QAgent(alpha=0.1, gamma=0.9, epsilon=-0.1)
