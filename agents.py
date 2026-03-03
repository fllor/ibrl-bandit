from main import *

class BaseAgent:
    def __init__(self, k : int, epsilon : float = 0.1, optimism : float = 0):
        self.num_actions = k
        self.k = k
        self.epsilon = epsilon
        self.optimism = optimism
        self.reset()
    def get_action(self): raise NotImplementedError
    def get_greedy_action(self): raise NotImplementedError
    def update(self, a, r, p): raise NotImplementedError


class BayesianThompsonAgent(BayesianAgent):
    def get_action(self):
        std_devs = 1.0 / np.sqrt(self.prec)
        samples = np.random.normal(self.values, std_devs)
        return samples.argmax()

class InfrabayesianThompsonAgent(InfrabayesianAgent):
    def get_action(self):
        diag_stds = 1.0 / np.sqrt(self.prec.diagonal())
        samples = np.random.normal(self.values.diagonal(), diag_stds)
        return samples.argmax()

class EXP3Agent(BaseAgent):
    def __init__(self, k, gamma=0.1, max_reward=1, optimism=0.0):
        self.gamma = gamma
        self.max_reward = max_reward
        self.eta = gamma / k
        super().__init__(k, epsilon=0, optimism=optimism)

    def reset(self):
        # We store weights in log-space for numerical stability
        # log(1.0) = 0
        self.log_weights = np.zeros(self.k)
        self.probs = np.ones(self.k) / self.k

    def get_action(self):
        # Sample based on probability distribution
        return np.random.choice(self.k, p=self.probs)

    def get_greedy_action(self):
        # The 'prediction' of the agent's intent is the highest weight
        return self.log_weights.argmax()

    def update(self, a, r, p=None):
            # 1. Internal scaling ensures the math stays in the [0, 1] 'safe zone'
            scaled_reward = r / self.max_reward
            
            # 2. Importance Sampling
            # This prevents the update from becoming too large if probs[a] is small
            estimated_reward = scaled_reward / self.probs[a]
            
            # 3. Update log-weights using the learning rate (eta)
            self.log_weights[a] += self.eta * estimated_reward
            
            # 4. Numerically stable softmax
            w = np.exp(self.log_weights - np.max(self.log_weights))
            self.probs = (1 - self.gamma) * (w / np.sum(w)) + (self.gamma / self.k)
            self.probs /= np.sum(self.probs)