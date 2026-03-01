import argparse
import numpy as np

class BanditEnvironment:
    def __init__(self, k=5):
        self.num_arms = k
        self.reset()
    def reset(self):
        self.true_values = np.random.normal(0, 1, (self.num_arms,))
    def interact(self, action, prediction=None):
        return np.random.normal(self.true_values[action], 1)
    def get_best_reward(self):
        return self.true_values.max()

class PolicyDependentBanditEnvironment(BanditEnvironment):
    def reset(self):
        # 2D matrix: [prediction][action]
        self.true_values = np.random.normal(0, 1, (self.num_arms, self.num_arms))
    def interact(self, action, prediction):
        return np.random.normal(self.true_values[prediction][action], 1)
    def get_best_reward(self):
        return self.true_values.diagonal().max()

class NewcombEnvironment(PolicyDependentBanditEnvironment):
    def __init__(self, k=2): # k is ignored here but kept for signature consistency
        super().__init__(k=2)
    def reset(self):
        self.true_values = np.array([[100, 101], [0, 1]])
    def interact(self, action, prediction):
        return self.true_values[prediction][action]
    def get_best_reward(self):
        return 100

class BaseAgent:
    def __init__(self, k, epsilon=0.1, optimism=5.0):
        self.k = k
        self.epsilon = epsilon
        self.optimism = optimism
        self.reset()
    def get_action(self): raise NotImplementedError
    def get_greedy_action(self): raise NotImplementedError
    def update(self, a, r, p): raise NotImplementedError

class ClassicalAgent(BaseAgent):
    def reset(self):
        self.values = np.ones(self.k) * self.optimism
        self.counts = np.zeros(self.k)
    def get_greedy_action(self): return self.values.argmax()
    def get_action(self):
        if np.random.rand() < self.epsilon: return np.random.randint(self.k)
        return self.get_greedy_action()
    def update(self, a, r, p=None):
        self.counts[a] += 1
        self.values[a] += (r - self.values[a]) / self.counts[a]

class BayesianAgent(BaseAgent):
    def reset(self):
        self.values = np.ones(self.k) * self.optimism
        self.prec = np.ones(self.k) * 0.1 
    def get_greedy_action(self): return self.values.argmax()
    def get_action(self):
        if np.random.rand() < self.epsilon: return np.random.randint(self.k)
        return self.get_greedy_action()
    def update(self, a, r, p=None):
        new_prec = self.prec[a] + 1.0
        self.values[a] = (self.prec[a] * self.values[a] + r) / new_prec
        self.prec[a] = new_prec

class InfrabayesianAgent(BaseAgent):
    def reset(self):
        self.values = np.ones((self.k, self.k)) * self.optimism
        self.prec = np.ones((self.k, self.k)) * 0.1
    def get_greedy_action(self): return self.values.diagonal().argmax()
    def get_action(self):
        if np.random.rand() < self.epsilon: return np.random.randint(self.k)
        return self.get_greedy_action()
    def update(self, a, r, p=0):
        # Default p=0 handles standard bandits that don't pass a prediction
        p = p if p is not None else 0
        new_prec = self.prec[p, a] + 1.0
        self.values[p, a] = (self.prec[p, a] * self.values[p, a] + r) / new_prec
        self.prec[p, a] = new_prec

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
