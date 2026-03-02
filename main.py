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


class NewcombEnvironment:
    """
    Newcomb's problem
    Two possible actions: 1-box or 2-box
    The reward depends on the predicted action and actual action of the agent

    Predicted 1-box & action 1-box -> reward 100
    Predicted 1-box & action 2-box -> reward 101
    Predicted 2-box & action 1-box -> reward 0
    Predicted 2-box & action 2-box -> reward 1
    """
    def __init__(self):
        self.true_values = np.array([
            [100,101],
            [0,1]
        ])

    def interact(self, action : int, prediction : int) -> float:
        assert action >= 0 and action < 2
        assert prediction >= 0 and prediction < 2
        return self.true_values[prediction][action]

    def get_best_action(self) -> int:
        return 0  # one-box

    def reset(self):
        pass


class QLearningAgent:
    """
    Classical Q-learning agent that interacts with a multi-armed bandit

    Arguments:
        learning_rate:  Learning rate for Q-learning
                        If None, use sample averages instead of Q-learning
        epsilon:        Parameter for epsilon-greedy policy encourage exploration
        optimism:       Initial Q-values to encourage early exploration
    """
    def __init__(self, k : int, learning_rate : float = 0.1, epsilon : float = 0.1, optimism : float = 0):
        self.num_actions = k
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.optimism = optimism

    def get_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.randint(0, self.num_actions)
        return self.values.argmax()

    def get_greedy_action(self):
        return self.values.argmax()

    def update(self, action : int, reward : float, prediction = None):
        if self.learning_rate is None:
            # Use sample average
            self.counts[action] += 1
            self.values[action] += (reward - self.values[action]) / self.counts[action]
        else:
            # Q-learning
            self.values[action] += self.learning_rate * (reward - self.values[action])

    def reset(self):
        if self.learning_rate is None:
            self.counts = np.zeros((self.num_actions,))
        self.values = np.ones((self.num_actions,))*self.optimism

def main(options):
    np.random.seed(options.seed)
    num_steps = options.steps
    num_runs = options.runs
    if options.environment == "bandit":
        env = BanditEnvironment(options.arms)
    elif options.environment == "newcomb":
        env = NewcombEnvironment()
        assert options.arms == 2
    else:
        raise RuntimeError("Invalid environment: " + options.environment)

    if options.agent.startswith("q"):
        agent = QLearningAgent(options.arms, options.learning, options.epsilon, options.optimism)
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
    parser.add_argument("-l", "--learning", help="Learning rate",               type=float,     default=0.1)
    parser.add_argument("-e", "--epsilon",  help="Parameter for exploration",   type=float,     default=0.1)
    parser.add_argument("-o", "--optimism", help="Parameter for reward priors", type=float,     default=0.0)
    parser.add_argument("-s", "--steps",    help="Number of steps per episode", type=int,       default=1000)
    parser.add_argument("-r", "--runs",     help="Number of episodes to run",   type=int,       default=1)
    parser.add_argument(      "--seed",     help="Random number seed",          type=int,       default=42)
    parser.add_argument("-v", "--verbose",  help="Debug output",                action="count", default=0)
    options = parser.parse_args()
    main(options)
