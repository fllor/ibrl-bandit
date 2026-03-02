import argparse
import math
import numpy as np

class BanditEnvironment:
    """
    Multi-armed bandit environment

    There are k discrete actions, each of which has a true values that is samples from a standard normal distribution
    The reward for a given action is sampled from a standard normal distribution shifted by the corresponding true value
    """
    def __init__(self, k : int):
        self.num_arms = k

    def interact(self, action : int, policy = None) -> float:
        assert action >= 0 and action < self.num_arms
        return np.random.normal(self.true_values[action], 1)

    def get_best_action(self) -> int:
        return self.true_values.argmax()

    def reset(self):
        self.true_values = np.random.normal(0, 1, (self.num_arms,))

class NewcombEnvironment:
    """
    Newcomb's problem

    Two actions:
      one-box (0): take box B
      two-box (1): take boxes A+B

    Box A is always filled with a small reward. Box B is filled with a larger reward, only if the agent the agent is predicted to one-box.
    """
    def __init__(self):
        self.boxA = 5   # guaranteed content of first box
        self.boxB = 10  # conditional content of second box

    def interact(self, action : int, policy) -> float:
        box_filled = np.random.binomial(1, policy[0])  # fill second box at one-boxing rate
        return self.boxA*(action==1) + self.boxB*box_filled

    def get_best_action(self) -> int:
        return 0  # one-box

    def reset(self):
        pass


class QLearningAgent:
    """
    Classical Q-learning agent that interacts with a multi-armed bandit

    Arguments:
        learning_rate:  Learning rate for Q-learning
    """
    def __init__(self, k : int, learning_rate : float = 0.1):
        self.num_actions = k
        self.learning_rate = learning_rate

    def get_policy(self):
        self.step += 1

        # Exploitation: randomly chose among the actions with highest value
        best_actions = self.q == self.q.max()
        exploit = np.ones((self.num_actions,))*best_actions / best_actions.sum()
        
        # Exploration: pick action uniformly
        explore = np.ones((self.num_actions,))/self.num_actions

        # epsilon-greedy policy with decaying epsilon
        epsilon = max(0.01, 0.5 / math.sqrt(self.step))
        return exploit * (1 - epsilon) + explore * epsilon

    def update(self, action : int, reward : float, prediction = None):
        self.q[action] += self.learning_rate * (reward - self.q[action])

    def reset(self):
        self.q = np.zeros((self.num_actions,))
        self.step = 0

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
        agent = QLearningAgent(options.arms, options.learning_rate)
    else:
        raise RuntimeError("Invalid agent type: " + options.agent)

    average_reward = np.zeros((num_steps,))
    average_reward_sq = np.zeros((num_steps,))
    best_action_freq = np.zeros((num_steps,))
    for r in range(num_runs):
        if options.verbose > 0:
            print(f"Run {r+1}/{num_runs}")
        env.reset()
        agent.reset()
        best_action = env.get_best_action()
        for i in range(num_steps):
            policy = agent.get_policy()
            policy /= policy.sum() # for numerics
            action = np.random.choice(len(policy), p=policy)
            reward = env.interact(action, policy)
            if options.verbose > 0:
                print(i, action, reward, policy.tolist(), agent.q.tolist())
            agent.update(action, reward, policy)
            average_reward[i] += reward
            average_reward_sq[i] += reward**2
            best_action_freq[i] += int(action == best_action)
    average_reward /= num_runs
    average_reward_sq /= num_runs
    best_action_freq /= num_runs

    for i in range(num_steps):
        print(
            i,
            average_reward[i],                                                          # Average reward
            math.sqrt(average_reward_sq[i] - average_reward[i]**2)/math.sqrt(num_runs), # Uncertainty of reward (converges to 0 as num_runs goes to infinity)
            math.sqrt(average_reward_sq[i] - average_reward[i]**2),                     # Spread of reward (converges to constant)
            best_action_freq[i],                                                        # Rate at which optimal action is taken
            math.sqrt(best_action_freq[i] - best_action_freq[i]**2)/math.sqrt(num_runs),# Uncertainty of optimal action rate
            math.sqrt(best_action_freq[i] - best_action_freq[i]**2)                     # Spread of optimal action rate
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL test with multi-armed bandit")
    parser.add_argument("environment",          help="Environment type",            type=str)
    parser.add_argument("agent",                help="Agent type",                  type=str)
    parser.add_argument("-k", "--arms",         help="Number of arms",              type=int,       default=10)
    parser.add_argument("-l", "--learning-rate",help="Learning rate",               type=float,     default=0.1)
    parser.add_argument("-s", "--steps",        help="Number of steps per episode", type=int,       default=1001)
    parser.add_argument("-r", "--runs",         help="Number of episodes to run",   type=int,       default=1)
    parser.add_argument(      "--seed",         help="Random number seed",          type=int,       default=42)
    parser.add_argument("-v", "--verbose",      help="Debug output",                action="count", default=0)
    options = parser.parse_args()
    main(options)
