from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import NDArray
import utils


class BaseAgent(ABC):
    """
    Base class for all agents
    """
    def __init__(self, num_actions : int):
        """
        Initialise permanent state
        Must call reset() before initial interaction with environment
        """
        assert num_actions >= 2
        self.num_actions = num_actions

    @abstractmethod
    def get_policy(self) -> NDArray[np.float64]:
        """
        Return the policy for the next episode.
        A policy is represented as a probability distribution of possible actions.
        """
        pass

    def update(self, policy : NDArray[np.float64], action : int, reward : float) -> None:
        """
        Update internal state based on outcome of the episode

        Arguments:
            policy: The policy chosen by the agent
            action: The action selected from the policy
            reward: The reward received
        """
        self.step += 1

    def reset(self) -> None:
        """
        Reset internal state
        Called before interacting with a new environment
        """
        self.step = 1


class GreedyAgent(BaseAgent):
    """
    Base class for agents that use either an epsilon-greedy or softmax policy to encourage exploration.

    Arguments:
        epsilon:        For epsilon-greedy policy
        temperature:    For softmax policy
        decay_type:     Select formula for decreasing rate (0: exponential, 1: linear)

    Both epsilon and temperature can be either a fixed float or a tuple (start,decay constant,min) for decreasing exploration.
    """
    def __init__(self, *args,
            epsilon     : float | tuple[float] | None = None,
            temperature : float | tuple[float] | None = None,
            decay_type  : float = 0,
            **kwargs):
        super().__init__(*args, **kwargs)

        if epsilon is not None and temperature is not None:
            raise RuntimeError("Cannot specify both epsilon and temperature")
        if epsilon is None and temperature is None:
            epsilon = 0.1  # default value

        assert epsilon is None or isinstance(epsilon,float) or (isinstance(epsilon,tuple) and len(epsilon)==3)
        assert temperature is None or isinstance(temperature,float) or (isinstance(temperature,tuple) and len(temperature)==3)

        self.epsilon = epsilon
        self.temperature = temperature
        self.decay_type = int(decay_type)
        assert self.decay_type in [0,1]

    def get_greedy_policy(self, values : NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Construct policy based on given reward estimates and selected algorithm
        """
        if self.epsilon is not None:
            return self.build_epsilon_greedy_policy(values)
        if self.temperature is not None:
            return self.build_softmax_policy(values)
        raise RuntimeError("Invalid state")

    def build_epsilon_greedy_policy(self, values : NDArray[np.float64]) -> NDArray[np.float64]:
        # Exploitation: sample uniformly across actions with highest value
        best_actions = (values == values.max())
        exploit = np.ones_like(values)*best_actions / best_actions.sum()

        # Exploration: sample uniformly across all actions
        explore = np.ones_like(values) / self.num_actions

        epsilon = self.parse_parameter(self.epsilon)
        return (1 - epsilon) * exploit + epsilon * explore

    def build_softmax_policy(self, values : NDArray[np.float64]) -> NDArray[np.float64]:
        temperature = self.parse_parameter(self.temperature)

        # Numerically stable softmax
        exp = np.exp((values - values.max()) / temperature)
        return exp / exp.sum()

    def parse_parameter(self, parameter : float | tuple[float]) -> float:
        """
        Parse exploration parameter (epsilon or temperature)
        Parameter may either be a fixed value or tuple specifying temporal decay
        """
        if isinstance(parameter,float):
            return parameter
        if self.decay_type == 0:  # exponential decay
            start,rate,end = parameter
            return max(start / (self.step ** rate), end)
        if self.decay_type == 1:  # linear decay
            start,last_step,end = parameter
            return end if self.step>=last_step else (start + (end-start) * (self.step / last_step))
        raise RuntimeError("Invalid state")


class QLearningAgent(GreedyAgent):
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

    def get_policy(self) -> NDArray[np.float64]:
        return self.get_greedy_policy(self.q)

    def update(self, policy : NDArray[np.float64], action : int, reward : float):
        super().update(policy, action, reward)
        self.q[action] += self.learning_rate * (reward - self.q[action])

    def reset(self):
        super().reset()
        self.q = np.zeros((self.num_actions,))


class BayesianAgent(GreedyAgent):
    """
    Agent using Bayesian inference

    For each action, keep track of a normal distribution that corresponds to the uncertainty of the associated reward.
    Update this estimate at each iteration based on the observed information.
    Picks the action with the largest expected reward.
    """
    def get_policy(self) -> NDArray[np.float64]:
        return self.get_greedy_policy(self.values)

    def update(self, policy : NDArray[np.float64], action : int, reward : float):
        super().update(policy, action, reward)
        # Estimate reward of the action and its uncertainty based on observed reward and priors
        # Define precision, tau, as 1/sigma^2 to avoid vanishing sigma precision issues
        self.values[action] = (self.precision[action] * self.values[action] + reward) / (self.precision[action] + 1.0)
        self.precision[action] += 1

    def reset(self):
        super().reset()
        self.values = np.zeros(self.num_actions)
        self.precision = np.ones(self.num_actions) * 0.1


class ExperimentalAgent1(QLearningAgent):
    """
    Instead of using the non-deterministic policy from Q-learning, sample an
    action from this policy and return a deterministic policy that chooses this
    action. Consequently, we only access the diagonal of the reward matrix.
    """
    def get_policy(self) -> NDArray[np.float64]:
        proto_policy = super().get_policy()
        action = utils.sample_action(proto_policy)
        policy = np.zeros((self.num_actions,))
        policy[action] = 1
        return policy


class ExperimentalAgent2(GreedyAgent):
    """
    Reconstruct full reward matrix

    This is achieved by sometimes picking a strongly peaked policy, such that
    we can be fairly certain what the predictor predicted. When we update, we
    know the actual action and the predicted action and are therefore able to
    update the correct entry of the reward matrix.
    Even with a strongly peaked policy, we will sometimes not chose the most
    likely action. This allows us to access the off-diagonal entries. As this
    rarely happens, learning the off-diagonal rewards is relatively slow. To
    speed up learning, we use sample averages, rather than Q-learning.

    When exploiting, we compute the optimal policy based on the current estimate
    of the reward matrix. This policy may be non-deterministic.
    """
    def __init__(self, *args,
            learning_rate : float = 0.1,
            **kwargs):
        assert "temperature" not in kwargs or kwargs["temperature"] is None
        super().__init__(*args, **kwargs)
        assert self.num_actions == 2  # technical limitation for now
        self.learning_rate = learning_rate
        self.update_threshold = 0.9 # minimum peak in policy to be considered for update
        self.exploration_peak = 20  # how strongly peaked should exploration policies be

    def get_policy(self) -> NDArray[np.float64]:
        epsilon = self.parse_parameter(self.epsilon)

        if np.random.binomial(1, epsilon):
            # exploration: pick a strongly peaked policy (with random peak)
            exploration = np.ones((self.num_actions,))
            exploration[np.random.randint(self.num_actions)] += self.exploration_peak
            exploration /= exploration.sum()
            return exploration
        else:
            # exploitation: compute optimal action based on current estimate of reward matrix
            (a,b),(c,d) = self.q.tolist()
            strategies = [ # (p(action0), expected reward)
                (1, a), # always pick action 0
                (0, d)  # always pick action 1
            ]
            if (a+d-b-c) < 0 and 0 < (b+c-2*d)/(b+c-a-d)/2 < 1: # mixed strategy
                strategies.append(((b+c-2*d)/(b+c-a-d)/2, (a*d-(b+c)**2/4)/(a+d-b-c)))
            p0 = max(strategies, key=lambda strategy: strategy[1])[0]
            return np.array([p0, 1-p0], dtype=np.float64)

    def update(self, policy : NDArray[np.float64], action : int, reward : float):
        super().update(policy, action, reward)
        prediction = policy.argmax()
        # only update action for which policy is strongly peaked, i.e. we can be fairly certain that the predictor chose this action
        if policy[prediction] < self.update_threshold:
            return
        # updates are weighted by the corresponding policy entry
        self.counts[prediction,action] += policy[prediction]
        self.q[prediction,action] += policy[prediction] * (reward - self.q[prediction,action]) / self.counts[prediction,action]
        #self.q[prediction,action] += policy[prediction] * self.learning_rate * (reward - self.q[prediction,action])

    def reset(self):
        super().reset()
        self.counts = np.zeros((self.num_actions,self.num_actions))
        self.q = np.zeros((self.num_actions,self.num_actions))


class EXP3Agent(BaseAgent):
    def __init__(self, num_actions : int, gamma : float = 0.1, max_reward : float = 1):
        self.num_actions = num_actions
        self.gamma = gamma
        self.max_reward = max_reward
        self.eta = gamma / num_actions

    def reset(self):
        # We store weights in log-space for numerical stability
        # log(1.0) = 0
        self.log_weights = np.zeros(self.num_actions)
        self.probs = np.ones(self.num_actions) / self.num_actions

    def get_policy(self):
        return self.probs

    def update(self, policy : NDArray[np.float64], action : int, reward : float):
        # 1. Internal scaling ensures the math stays in the [0, 1] 'safe zone'
        scaled_reward = reward / self.max_reward
        
        # 2. Importance Sampling
        # This prevents the update from becoming too large if probs[a] is small
        estimated_reward = scaled_reward / self.probs[action]
        
        # 3. Update log-weights using the learning rate (eta)
        self.log_weights[action] += self.eta * estimated_reward
        
        # 4. Numerically stable softmax
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        self.probs = (1 - self.gamma) * (weights / np.sum(weights)) + (self.gamma / self.num_actions)
        self.probs /= self.probs.sum()
