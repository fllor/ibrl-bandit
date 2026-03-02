import argparse
import numpy as np

class BanditEnvironment:
    """
    Multi-armed bandit environment

    There are k discrete actions, each of which has a true values that is samples from a standard normal distribution
    The reward for a given action is sampled from a standard normal distribution shifted by the corresponding true value
    """
    def __init__(self, k : int):
        self.num_arms = k
        self.reset()

    def interact(self, action : int, prediction = None) -> float:
        assert action >= 0 and action < self.num_arms
        return np.random.normal(self.true_values[action], 1)

    def get_best_action(self) -> int:
        return self.true_values.argmax()

    def get_best_reward(self):
        return self.true_values.max()

    def reset(self):
        self.true_values = np.random.normal(0, 1, (self.num_arms,))

class PolicyDependentBanditEnvironment:
    """
    Like a multi-armed bandit, but the reward depends on the (prediction,action) pair, rather than just the action
    """
    def __init__(self, k : int, sample : bool = True):
        self.num_arms = k
        self.sample = sample
        self.reset()

    def interact(self, action : int, prediction : int) -> float:
        assert action >= 0 and action < self.num_arms
        assert prediction >= 0 and prediction < self.num_arms
        if self.sample:
            return np.random.normal(self.true_values[prediction][action], 1)
        else:
            return self.true_values[prediction][action]

    def get_best_action(self) -> int:
        # Logically consistent actions are along the diagonal
        return self.true_values.diagonal().argmax()

    def get_best_reward(self) -> int:
        # Logically consistent actions are along the diagonal
        return self.true_values.diagonal().max()

    def reset(self):
        self.true_values = np.random.normal(0, 1, (self.num_arms,self.num_arms))  # 2D array

class NewcombEnvironment(PolicyDependentBanditEnvironment):
    """
    Newcomb's problem (special case of policy-dependent bandit)
    Two possible actions: 1-box or 2-box
    The reward depends on the predicted action and actual action of the agent

    Predicted 1-box & action 1-box -> reward 100
    Predicted 1-box & action 2-box -> reward 101
    Predicted 2-box & action 1-box -> reward 0
    Predicted 2-box & action 2-box -> reward 1
    """
    def __init__(self, k : int = 2): # k is ignored here but kept for signature consistency
        super().__init__(k=2)

    def reset(self):
        self.true_values = np.array([
            [100,101],
            [0,1]
        ])
        
    def interact(self, action, prediction):
        return self.true_values[prediction][action]
    def get_best_reward(self):
        return 100


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

class ClassicalAgent(BaseAgent):
    """
    Classical reinforcement learning agent that interacts with a multi-armed bandit

    Action values are estimated as sample averages (which assumes true values are stationary)
    Use epsilon-greedy policy to balance continuous exploration
    Optionally start with optimism to encourage early exploration
    """

    def get_action(self):
        if np.random.rand() < self.epsilon: return np.random.randint(self.k)
        return self.get_greedy_action()

    def get_greedy_action(self):
        return self.values.argmax()

    def update(self, action : int, reward : float, prediction = None):
        # Update sample averages based on new observation
        # Equation 2.3 from Barto&Sutton
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

    def reset(self):
        self.values = np.ones(self.k) * self.optimism
        self.counts = np.zeros(self.k)

class BayesianAgent(BaseAgent):
    """
    Agent using Bayesian inference

    For each action it estimates a normal distribution and the picks the action with the largest central value
    Use epsilon-greedy policy to balance continuous exploration
    Optionally start with optimism to encourage early exploration
    """
    def reset(self):
        self.values = np.ones(self.k) * self.optimism
        self.prec = np.ones(self.k) * 0.1 
    
    def get_action(self):
        if np.random.rand() < self.epsilon: 
            return np.random.randint(self.k)
        return self.get_greedy_action()
    
    def get_greedy_action(self): 
        return self.values.argmax()

    def update(self, a: int, r: float, p=None):
        new_prec = self.prec[a] + 1.0
        self.values[a] = (self.prec[a] * self.values[a] + r) / new_prec
        self.prec[a] = new_prec

class InfrabayesianAgent(BaseAgent):

    def get_action(self):
        if np.random.rand() < self.epsilon: 
            return np.random.randint(self.k)
        return self.get_greedy_action()

    def get_greedy_action(self):
        # We have expected values for all (prediction,action) pairs.
        # However, all entries with prediction!=action are logically inconsistent. They get sent to nirvana,
        # by assigning them infinite reward. For each action, we then take the minimum possible reward over
        # all predictions (environments).
        # Effectively, this means we just need to consider the diagonal of the expected-value matrix
        return self.values.diagonal().argmax()

    def update(self, a, r, p=0):
        # Default p=0 handles standard bandits that don't pass a prediction
        p = p if p is not None else 0
        new_prec = self.prec[p, a] + 1.0
        self.values[p, a] = (self.prec[p, a] * self.values[p, a] + r) / new_prec
        self.prec[p, a] = new_prec

    def reset(self):
        self.values = np.ones((self.k, self.k)) * self.optimism
        self.prec = np.ones((self.k, self.k)) * 0.1

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

def main(options):
    np.random.seed(options.seed)
    num_steps = options.steps
    num_runs = options.runs
    if options.environment == "bandit":
        env = BanditEnvironment(options.arms)
    elif options.environment == "pdbandit":
        env = PolicyDependentBanditEnvironment(options.arms)
    elif options.environment == "newcomb":
        env = NewcombEnvironment()
        assert options.arms == 2
    else:
        raise RuntimeError("Invalid environment: " + options.environment)

    if options.agent.startswith("classic"):
        agent = ClassicalAgent(options.arms, options.epsilon, options.optimism)
    elif options.agent.startswith("bayes"):
        agent = BayesianAgent(options.arms, options.epsilon, options.optimism)
    elif options.agent.startswith("infrabayes"):
        agent = InfrabayesianAgent(options.arms, options.epsilon, options.optimism)
    else:
        raise RuntimeError("Invalid agent type: " + options.agent)

    average_reward = np.zeros((num_steps,))
    best_action_freq = np.zeros((num_steps,))
    for r in range(num_runs):
        if options.verbose > 0:
            print(f"Run {r+1}/{num_runs}")
        env.reset()
        agent.reset()
        best_action = env.get_best_action()
        for i in range(num_steps):
            action = agent.get_action()         # the actual action (might be random)
            greedy = agent.get_greedy_action()  # most likely action (what would be predicted)
            reward = env.interact(action, greedy)
            agent.update(action, reward, greedy)
            average_reward[i] += reward
            best_action_freq[i] += int(action == best_action)
            if options.verbose > 0:
                print(i, action, greedy, reward, agent.values.tolist())
    average_reward /= num_runs
    best_action_freq /= num_runs

    for i in range(num_steps):
        print(i, average_reward[i], best_action_freq[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL test with multi-armed bandit")
    parser.add_argument("environment",      help="Environment type",            type=str)
    parser.add_argument("agent",            help="Agent type",                  type=str)
    parser.add_argument("-k", "--arms",     help="Number of arms",              type=int,       default=10)
    parser.add_argument("-e", "--epsilon",  help="Parameter for exploration",   type=float,     default=0.1)
    parser.add_argument("-o", "--optimism", help="Parameter for reward priors", type=float,     default=0.0)
    parser.add_argument("-s", "--steps",    help="Number of steps per episode", type=int,       default=1000)
    parser.add_argument("-r", "--runs",     help="Number of episodes to run",   type=int,       default=1)
    parser.add_argument(      "--seed",     help="Random number seed",          type=int,       default=42)
    parser.add_argument("-v", "--verbose",  help="Debug output",                action="count", default=0)
    options = parser.parse_args()
    main(options)
