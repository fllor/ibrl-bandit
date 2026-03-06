import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent


class ExperimentalAgent2(BaseGreedyAgent):
    """
    Reconstruct full reward matrix

    This is achieved by sometimes picking a strongly peaked probability distribution, such that we can be fairly
    certain about what the predictor predicted. When we update, we know the actual action and the predicted action
    and are therefore able to update the correct entry of the reward matrix.
    Even with a strongly peaked distribution, we will sometimes not chose the most likely action. This allows us to
    access the off-diagonal entries. As this rarely happens, learning the off-diagonal rewards is relatively slow. To
    speed up learning, we use sample averages, rather than Q-learning.

    When exploiting, we compute the optimal distribution based on the current estimate of the reward matrix. This
    distribution may be non-deterministic.
    """
    def __init__(self, *args,
            learning_rate : float = 0.1,
            **kwargs):
        assert "temperature" not in kwargs or kwargs["temperature"] is None
        super().__init__(*args, **kwargs)
        assert self.num_actions == 2  # technical limitation for now
        self.learning_rate = learning_rate
        self.update_threshold = 0.9 # minimum probability to be considered for update
        self.exploration_peak = 20  # how strongly peaked should exploration policies be

    def get_probabilities(self) -> NDArray[np.float64]:
        epsilon = self.parse_parameter(self.epsilon)

        if self.random.binomial(1, epsilon):
            # exploration: pick a strongly peaked distribution (with random peak)
            exploration = np.ones((self.num_actions,))
            exploration[self.random.integers(self.num_actions)] += self.exploration_peak
            exploration /= exploration.sum()
            return exploration
        else:
            # exploitation: compute optimal action based on current estimate of reward matrix
            (a,b),(c,d) = self.q.tolist()
            strategies = [ # (p(action0), expected reward)
                (1, a), # always pick action 0
                (0, d)  # always pick action 1
            ]
            if (a+d-b-c) < 0 and (b+c-a-d) != 0 and 0 < (b+c-2*d)/(b+c-a-d)/2 < 1: # mixed strategy
                strategies.append(((b+c-2*d)/(b+c-a-d)/2, (a*d-(b+c)**2/4)/(a+d-b-c)))
            p0 = max(strategies, key=lambda strategy: strategy[1])[0]
            return np.array([p0, 1-p0], dtype=np.float64)

    def update(self, probabilities : NDArray[np.float64], action : int, reward : float):
        super().update(probabilities, action, reward)
        prediction = probabilities.argmax()
        # only update action for which distribution is strongly peaked,
        # i.e. when we can be fairly certain that the predictor chose this action
        if probabilities[prediction] < self.update_threshold:
            return
        # updates are weighted by the corresponding probability
        self.counts[prediction,action] += probabilities[prediction]
        self.q[prediction,action] += probabilities[prediction] * (reward - self.q[prediction,action]) / self.counts[prediction,action]
        #self.q[prediction,action] += probabilities[prediction] * self.learning_rate * (reward - self.q[prediction,action])

    def reset(self):
        super().reset()
        self.counts = np.zeros((self.num_actions,self.num_actions))
        self.q = np.zeros((self.num_actions,self.num_actions))
