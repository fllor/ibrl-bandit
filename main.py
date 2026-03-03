import argparse
import math
from enum import Enum
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

    def get_optimal_reward(self) -> int:
        return self.true_values.max()

    def reset(self):
        self.true_values = np.random.normal(0, 1, (self.num_arms,))

class NewcombLikeEnvironment:
    """
    Policy dependent environment with two possible actions

    If action i was predicted and action j was taken, the reward will be reward_table[i][j]
    """
    def __init__(self, reward_table):
        assert len(reward_table) == 2
        self.reward_table = reward_table

    def interact(self, action : int, policy) -> float:
        prediction = np.random.choice(len(policy), p=policy)
        return self.reward_table[prediction][action]

    def get_optimal_reward(self) -> int:
        # The reward is a quadratic function of the probability of taking action 0.
        # Thus, there are three policies that could potentially be optimal
        (a,b),(c,d) = self.reward_table
        return max(
            a,  # always take action 0
            d,  # always take action 1
            (a*d-(b+c)**2/4)/(a+d-b-c) if (a+d-b-c) != 0 else float("-inf")
                # take action 0 with probability (b+c-2*d)/(b+c-a-d)/2
        )

    def reset(self):
        pass


class NewcombEnvironment(NewcombLikeEnvironment):
    def __init__(self):
        boxA = 5   # guaranteed content of first box
        boxB = 10  # conditional content of second box
        super().__init__([
            [boxB, boxB+boxA],
            [0,    boxA     ]
        ])

class DeathInDamascusEnvironment(NewcombLikeEnvironment):
    def __init__(self, asymmetry = 0.):
        death = 0  # reward upon death
        life = 10  # reward upon survival
        super().__init__([
            [death, life ],
            [life,  death],
        ])

class AsymmetricDeathInDamascusEnvironment(NewcombLikeEnvironment):
    def __init__(self):
        death_in_damascus = 0   # reward upon death in Damascus
        death_in_aleppo = 5     # reward upon death in Aleppo
        life = 10               # reward upon survival
        super().__init__([
            [death_in_damascus, life           ],
            [life,              death_in_aleppo],
        ])

class CoordinationGameEnvironment(NewcombLikeEnvironment):
    def __init__(self):
        super().__init__([
            [2, 0],
            [0, 1],
        ])

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

class QLearningAgent:
    """
    Classical Q-learning agent that interacts with a multi-armed bandit

    Arguments:
        policy_function: Function to build policy from Q-values
        learning_rate:   Learning rate for Q-learning
    """
    def __init__(self, k : int, policy_function, learning_rate : float = 0.1):
        self.num_actions = k
        self.policy_function = policy_function
        self.learning_rate = learning_rate

    def get_policy(self):
        self.step += 1
        return self.policy_function(self.q, self.num_actions, self.step)

    def update(self, action : int, reward : float, prediction = None):
        self.q[action] += self.learning_rate * (reward - self.q[action])

    def reset(self):
        self.q = np.zeros((self.num_actions,))
        self.step = 0

class ExperimentalAgent1(QLearningAgent):
    """
    Instead of using the non-deterministic policy from Q-learning, sample an
    action from this policy and return a deterministic policy that chooses this
    action. Consequently, we only access the diagonal of the reward matrix.
    """
    def get_policy(self):
        proto_policy = super().get_policy()
        proto_policy /= proto_policy.sum() # for numerics
        action = np.random.choice(len(proto_policy), p=proto_policy)
        policy = np.zeros((self.num_actions,))
        policy[action] = 1
        return policy

def main(options):
    np.random.seed(options.seed)
    num_steps = options.steps
    num_runs = options.runs
    if options.environment == "bandit":
        env = BanditEnvironment(options.arms)
    elif options.environment == "newcomb":
        env = NewcombEnvironment()
        assert options.arms == 2
    elif options.environment == "damascus":
        env = DeathInDamascusEnvironment()
        assert options.arms == 2
    elif options.environment == "asymmetric-damascus":
        env = AsymmetricDeathInDamascusEnvironment()
        assert options.arms == 2
    elif options.environment == "coordination":
        env = CoordinationGameEnvironment()
        assert options.arms == 2
    else:
        raise RuntimeError("Invalid environment: " + options.environment)

    if options.policy.startswith("epsilon"):
        policy_function = epsilon_greedy
    elif options.policy.startswith("softmax"):
        policy_function = softmax
    else:
        raise RuntimeError("Invalid policy type: " + options.agent)

    if options.agent.startswith("classical"):
        agent = QLearningAgent(options.arms, policy_function, options.learning_rate)
    elif options.agent.startswith("experimental1"):
        agent = ExperimentalAgent1(options.arms, policy_function, options.learning_rate)
    else:
        raise RuntimeError("Invalid agent type: " + options.agent)

    average_reward = np.zeros((num_steps,))
    average_reward_sq = np.zeros((num_steps,))
    best_reward = 0
    for r in range(num_runs):
        if options.verbose > 0:
            print(f"Run {r+1}/{num_runs}")
        env.reset()
        agent.reset()
        best_reward += env.get_optimal_reward()
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

    average_reward /= num_runs
    average_reward_sq /= num_runs
    average_reward_spread = np.sqrt(average_reward_sq - average_reward**2)
    average_reward_unc = average_reward_spread / np.sqrt(num_runs)
    best_reward /= num_runs

    for i in range(num_steps):
        print(
            i,
            best_reward,                # Best possible reward (as determined by environment)
            average_reward[i],          # Average reward received by agent
            average_reward_unc[i],      # Uncertainty of reward (converges to 0 as num_runs goes to infinity)
            average_reward_spread[i],   # Spread of reward (converges to constant)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL test with multi-armed bandit")
    parser.add_argument("environment",          help="Environment type",            type=str)
    parser.add_argument("agent",                help="Agent type",                  type=str)
    parser.add_argument("-p", "--policy",       help="Method to build policy",      type=str,       default="epsilon-greedy")
    parser.add_argument("-k", "--arms",         help="Number of arms",              type=int,       default=2)
    parser.add_argument("-l", "--learning-rate",help="Learning rate",               type=float,     default=0.1)
    parser.add_argument("-s", "--steps",        help="Number of steps per episode", type=int,       default=1001)
    parser.add_argument("-r", "--runs",         help="Number of episodes to run",   type=int,       default=1)
    parser.add_argument(      "--seed",         help="Random number seed",          type=int,       default=42)
    parser.add_argument("-v", "--verbose",      help="Debug output",                action="count", default=0)
    options = parser.parse_args()
    main(options)
