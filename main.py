import argparse
import numpy as np

class BanditEnv:
    """
    Multi-armed bandit environment (stateless)

    There are k discrete actions, each of which has a true values that is samples from a standard normal distribution
    The reward for a given action is sampled from a standard normal distribution shifted by the corresponding true value
    """
    def __init__(self, k : int):
        self.num_arms = k
        self.true_values = np.random.normal(0, 1, (k,))

    def interact(self, action : int) -> float:
        assert action >= 0 and action < self.num_arms
        return np.random.normal(self.true_values[action], 1)

    def get_best_action(self) -> int:
        return self.true_values.argmax()

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
        self.values = np.ones((k,))*optimism
        self.counts = np.zeros((k,))

    def get_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.randint(0, self.num_actions)
        return self.values.argmax()

    def update(self, action : int, reward : float):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

class BayesianAgent:
    """
    Agent using Bayesian inference

    For each action it estimates a normal distribution and the picks the action with the largest central value
    Use epsilon-greedy policy to balance continuous exploration
    """
    def __init__(self, k : int, epsilon : float = 0.1):
        self.num_actions = k
        self.epsilon = epsilon
        self.sigma_true = 1             # known standard deviation
        self.values = np.zeros((k,))    # prior: central value 0
        self.sigma = np.ones((k,))*self.sigma_true

    def get_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.randint(0, self.num_actions)
        return self.values.argmax()

    def update(self, action : int, reward : float):
        tmp = 1/self.sigma[action]**2 + 1/self.sigma_true**2
        self.values[action] = (self.values[action]/self.sigma[action]**2 + reward/self.sigma_true**2)/tmp
        self.sigma[action] = 1/np.sqrt(tmp)

def main(options):
    np.random.seed(42)
    k = options.arms
    epsilon = options.epsilon
    num_steps = options.steps
    num_runs = options.runs
    if options.agent.startswith("classic"):
        Agent = ClassicalAgent
    elif options.agent.startswith("bayes"):
        Agent = BayesianAgent
    else:
        raise RuntimeError("Invalid agent type: " + options.agent)

    average_reward = np.zeros((num_steps,))
    best_action_freq = np.zeros((num_steps,))
    for _ in range(num_runs):
        env = BanditEnv(k)
        agent = Agent(k, epsilon)
        best_action = env.get_best_action()
        for i in range(num_steps):
            action = agent.get_action()
            reward = env.interact(action)
            agent.update(action, reward)
            average_reward[i] += reward
            best_action_freq[i] += int(action == best_action)
    average_reward /= num_runs
    best_action_freq /= num_runs

    for i in range(num_steps):
        print(i, average_reward[i], best_action_freq[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL test with multi-armed bandit")
    parser.add_argument("agent", help="Agent type", type=str)
    parser.add_argument("-k", "--arms", help="Number of arms", type=int, default=10)
    parser.add_argument("-e", "--epsilon", help="Parameter of epsilon-greedy policy", type=float, default=0.1)
    parser.add_argument("-o", "--optimism", help="Parameter for classical agent", type=float, default=0.0)
    parser.add_argument("-s", "--steps", help="Number of steps to run", type=int, default=1000)
    parser.add_argument("-r", "--runs", help="Number of episodes to simulate", type=int, default=1)
    options = parser.parse_args()
    main(options)
