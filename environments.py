from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import NDArray
import utils


class BaseEnvironment(ABC):
    @abstractmethod
    def interact(self, action : int, policy : NDArray[np.float64] = None) -> float:
        """
        Let the predictor act, based on the given policy.
        Compute the reward corresponding to the agent's action.
        (Conceptually, these are two distinct steps)

        Arguments:
            policy: Policy chosen by the agent
            action: Action sampled from the policy
        
        Returns:
            reward of the interaction
        """
        pass

    @abstractmethod
    def get_optimal_reward(self) -> float:
        """
        Compute the average reward obtained by the optimal policy

        Returns:
            average reward of optimal policy
        """
        pass

    def reset(self):
        """
        Reset internal state. Potentially initialise randomly
        """
        pass


class BanditEnvironment(BaseEnvironment):
    """
    Multi-armed bandit environment

    There are k discrete actions, each of which has a true values that is samples from a standard normal distribution
    The reward for a given action is sampled from a standard normal distribution shifted by the corresponding true value
    """
    def __init__(self, num_arms : int):
        self.num_arms = num_arms
        self.seed = 42

    def interact(self, action : int, policy : NDArray[np.float64]) -> float:
        assert action >= 0 and action < self.num_arms
        return np.random.normal(self.values[action], 1)

    def get_optimal_reward(self) -> int:
        return self.values.max()

    def reset(self):
        # Seed separate RNG, such that the environment is consistent across agents
        self.seed += 1
        self.random = np.random.default_rng(self.seed)
        self.values = self.random.normal(0, 1, (self.num_arms,))


class NewcombLikeEnvironment(BaseEnvironment):
    """
    Policy dependent environment with two possible actions

    If action i was predicted and action j was taken, the reward will be reward_table[i,j]
    """
    def __init__(self, reward_table):
        assert len(reward_table) == 2
        self.reward_table = np.array(reward_table)

    def interact(self, action : int, policy) -> float:
        prediction = utils.sample_action(policy)
        return self.reward_table[prediction,action]

    def get_optimal_reward(self) -> int:
        # The reward is a quadratic function of the probability of taking action 0.
        # Thus, there are three policies that could potentially be optimal
        (a,b),(c,d) = self.reward_table.tolist()
        return max(
            a,  # always take action 0
            d,  # always take action 1
            (a*d-(b+c)**2/4)/(a+d-b-c) if (a+d-b-c) < 0 else float("-inf")
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


class PolicyDependentBanditEnvironment(NewcombLikeEnvironment):
    """
    Like a multi-armed bandit, but the reward depends on the (prediction,action) pair, rather than just the action.
    This is a generalisation of both the multi-armed bandit and Newcomb-like problems
    """
    def __init__(self):
        self.num_arms = 2
        self.seed = 42
        super().__init__(self.sample_reward_table())

    def reset(self):
        self.reward_table = self.sample_reward_table()

    def sample_reward_table(self) -> NDArray[np.float64]:
        # Seed separate RNG, such that the environment is consistent across agents
        self.seed += 1
        self.random = np.random.default_rng(self.seed)
        return self.random.normal(0, 1, (self.num_arms,self.num_arms))  # 2D array

