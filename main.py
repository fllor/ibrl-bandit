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

    def interact(self, action : int, prediction = None) -> float:
        assert action >= 0 and action < self.num_arms
        return np.random.normal(self.true_values[action], 1)

    def get_best_action(self) -> int:
        return self.true_values.argmax()

    def reset(self):
        self.true_values = np.random.normal(0, 1, (self.num_arms,))

class PolicyDependentBanditEnvironment:
    """
    Like a multi-armed bandit, but the reward depends on the (prediction,action) pair, rather than just the action
    """
    def __init__(self, k : int):
        self.num_arms = k

    def interact(self, action : int, prediction : int) -> float:
        assert action >= 0 and action < self.num_arms
        assert prediction >= 0 and prediction < self.num_arms
        return np.random.normal(self.true_values[prediction][action], 1)

    def get_best_action(self) -> int:
        # Logically consistent actions are along the diagonal
        return self.true_values.diagonal().argmax()

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
    def __init__(self):
        super().__init__(2)

    def reset(self):
        self.true_values = np.array([
            [100,101],
            [0,1]
        ])


class ClassicalAgent:
    """
    Classical reinforcement learning agent that interacts with a multi-armed bandit

    Action values are estimated as sample averages (which assumes true values are stationary)
    Use epsilon-greedy policy to balance continuous exploration
    Optionally start with optimism to encourage early exploration
    """
    def __init__(self, k : int, epsilon : float = 0.1, optimism : float = 0):
        self.num_actions = k
        self.epsilon = epsilon
        self.optimism = optimism

    def get_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.randint(0, self.num_actions)
        return self.values.argmax()

    def get_greedy_action(self):
        return self.values.argmax()

    def update(self, action : int, reward : float, prediction = None):
        # Update sample averages based on new observation
        # Equation 2.3 from Barto&Sutton
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

    def reset(self):
        self.values = np.ones((self.num_actions,))*self.optimism
        self.counts = np.zeros((self.num_actions,))

class BayesianAgent:
    """
    Agent using Bayesian inference

    For each action it estimates a normal distribution and the picks the action with the largest central value
    Use epsilon-greedy policy to balance continuous exploration
    Optionally start with optimism to encourage early exploration
    """
    def __init__(self, k : int, epsilon : float = 0.1, optimism : float = 0):
        self.num_actions = k
        self.epsilon = epsilon
        self.optimism = optimism
        self.sigma_true = 1 # assume standard deviation of reward sampling is known

    def get_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.randint(0, self.num_actions)
        return self.values.argmax()

    def get_greedy_action(self):
        return self.values.argmax()

    def update(self, action : int, reward : float, prediction = None):
        # estimate reward of the action and its uncertainty based on observed reward and priors
        # Formulae from https://slinderman.github.io/stats305c/notebooks/01_bayes_normal.html#normal-model-with-unknown-mean
        tmp = 1/self.sigma[action]**2 + 1/self.sigma_true**2
        self.values[action] = (self.values[action]/self.sigma[action]**2 + reward/self.sigma_true**2)/tmp
        self.sigma[action] = 1/np.sqrt(tmp)

    def reset(self):
        self.values = np.ones((self.num_actions,))*self.optimism   # prior for rewards, will converge to true values
        self.sigma = np.ones((self.num_actions,))*self.sigma_true  # uncertainty of rewards, will converge to 0

class InfrabayesianAgent:
    def __init__(self, k : int, epsilon : float = 0.1, optimism : float = 0):
        self.num_actions = k
        self.epsilon = epsilon
        self.optimism = optimism
        self.sigma_true = 1 # assume standard deviation of reward sampling is known

    def get_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.randint(0, self.num_actions)
        return self.get_greedy_action()

    def get_greedy_action(self):
        # We have expected values for all (prediction,action) pairs.
        # However, all entries with prediction!=action are logically inconsistent. They get sent to nirvana,
        # by assigning them infinite reward. For each action, we then take the minimum possible reward over
        # all predictions (environments).
        # Effectively, this means we just need to consider the diagonal of the expected-value matrix
        return self.values.diagonal().argmax()

    def update(self, action : int, reward : float, prediction : int):
        # estimate reward of the action and its uncertainty based on observed reward and priors
        tmp = 1/self.sigma[prediction][action]**2 + 1/self.sigma_true**2
        self.values[prediction][action] = (self.values[prediction][action]/self.sigma[prediction][action]**2 + reward/self.sigma_true**2)/tmp
        self.sigma[prediction][action] = 1/np.sqrt(tmp)

    def reset(self):
        # 2D array to hold expected reward for all (prediction,action) pairs:
        self.values = np.ones((self.num_actions,self.num_actions))*self.optimism   # prior for rewards, will converge to true values
        self.sigma = np.ones((self.num_actions,self.num_actions))*self.sigma_true  # uncertainty of rewards, will converge to 0

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
