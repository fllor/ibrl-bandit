from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import NDArray
import utils


def epsilon_greedy(q, num_actions, step):
    # Exploitation: uniformly chose among the actions with highest value
    best_actions = (q == q.max())
    exploit = np.ones((num_actions,))*best_actions / best_actions.sum()

    # Exploration: uniformly pick any action
    explore = np.ones((num_actions,))/num_actions

    # epsilon-greedy policy with decaying epsilon
    epsilon = max(0.01, 0.5 / (step ** 0.5))
    return exploit * (1 - epsilon) + explore * epsilon

def softmax(q, num_actions, step):
    # Softmax of Q-values with decaying temperature
    temperature = max(0.05, 1.0 / (step ** 0.3))

    # Numerically stable softmax
    q_shifted = q - q.max()
    exp_q = np.exp(q_shifted / temperature)
    return exp_q / exp_q.sum()


class BaseAgent(ABC):
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
        pass

    def reset(self) -> None:
        """
        Reset internal state
        """
        pass


class QLearningAgent(BaseAgent):
    """
    Classical Q-learning agent that interacts with a multi-armed bandit

    Arguments:
        num_actions:     Number of possible action in environment
        policy_function: Function to build policy from Q-values
        learning_rate:   Learning rate for Q-learning
    """
    def __init__(self, num_actions : int, policy_function : callable, learning_rate : float = 0.1):
        self.num_actions = num_actions
        self.policy_function = policy_function
        self.learning_rate = learning_rate

    def get_policy(self) -> NDArray[np.float64]:
        self.step += 1
        return self.policy_function(self.q, self.num_actions, self.step)

    def update(self, policy : NDArray[np.float64], action : int, reward : float):
        self.q[action] += self.learning_rate * (reward - self.q[action])

    def reset(self):
        self.q = np.zeros((self.num_actions,))
        self.step = 0


class BayesianAgent(BaseAgent):
    """
    Agent using Bayesian inference

    For each action it estimates a normal distribution and the picks the action with the largest central value
    Use epsilon-greedy policy to balance continuous exploration
    Optionally start with optimism to encourage early exploration
    """
    def __init__(self, num_actions : int, policy_function : callable):
        self.num_actions = num_actions
        self.policy_function = policy_function

    def get_policy(self) -> NDArray[np.float64]:
        self.step += 1
        return self.policy_function(self.values, self.num_actions, self.step)

    def update(self, policy : NDArray[np.float64], action : int, reward : float):
        # Estimate reward of the action and its uncertainty based on observed reward and priors
        # Define precision, tau, as 1/sigma^2 to avoid vanishing sigma precision issues
        self.values[action] = (self.precision[action] * self.values[action] + reward) / (self.precision[action] + 1.0)
        self.precision[action] += 1

    def reset(self):
        self.step = 0
        self.values = np.zeros(self.num_actions)
        self.precision = np.ones(self.num_actions) * 0.1


#class BayesianThompsonAgent(BayesianAgent):
#    def get_policy(self) -> NDArray[np.float64]:
#        std_devs = 1.0 / np.sqrt(self.precision)
#        samples = np.random.normal(self.values, std_devs)
#        return samples.argmax()


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


class ExperimentalAgent2(BaseAgent):
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
    def __init__(self, k : int, learning_rate : float = 0.1):
        assert k == 2  # technical limitation for now
        self.num_actions = k
        self.learning_rate = learning_rate
        self.update_threshold = 0.9 # minimum peak in policy to be considered for update
        self.exploration_peak = 20  # how strongly peaked should exploration policies be

    def get_policy(self) -> NDArray[np.float64]:
        self.step += 1
        #epsilon = max(0.01, 0.5 / (self.step ** 0.5))  # default
        epsilon = max(0.01, 0.5 - 0.49 * self.step/700)  # schedule with more exploration
        #epsilon = 0.1  # fixed exploration

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
        prediction = policy.argmax()
        # only update action for which policy is strongly peaked, i.e. we can be fairly certain that the predictor chose this action
        if policy[prediction] < self.update_threshold:
            return
        # updates are weighted by the corresponding policy entry
        self.counts[prediction,action] += policy[prediction]
        self.q[prediction,action] += policy[prediction] * (reward - self.q[prediction,action]) / self.counts[prediction,action]
        #self.q[prediction,action] += policy[prediction] * self.learning_rate * (reward - self.q[prediction,action])

    def reset(self):
        self.counts = np.zeros((self.num_actions,self.num_actions))
        self.q = np.zeros((self.num_actions,self.num_actions))
        self.step = 0


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
