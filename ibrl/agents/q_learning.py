import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent


class QLearningAgent(BaseGreedyAgent):
    """
    Classical Q-learning agent that interacts with a multi-armed bandit

    Arguments:
        learning_rate:  Learning rate for Q-learning
    """
    def __init__(self, *args,
            learning_rate : float = 0.1,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate

    def get_probabilities(self) -> NDArray[np.float64]:
        return self.build_greedy_policy(self.q)

    def update(self, probabilities : NDArray[np.float64], action : int, reward : float):
        super().update(probabilities, action, reward)
        self.q[action] += self.learning_rate * (reward - self.q[action])

    def reset(self):
        super().reset()
        self.q = np.zeros((self.num_actions,))
