from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import NDArray
import utils


class BaseEnvironment(ABC):
    """
    Base class for all environments

    Assumes a finite number of discrete actions

    Arguments:
        num_actions: Number of discrete actions
        num_steps:   Number of steps per run (for planning)
        num_runs:    Number of runs (for planning)
        seed:        Seed for random number generator
        verbose:     Request debugging output
    """
    def __init__(self,
            num_actions : int,
            num_steps : int = None,
            num_runs : int = None,
            *,
            seed : int = 0x89abcdef,  # Default needs to be different from agent
            verbose : int = 0):
        """
        Initialise permanent state
        Must call reset() before initial interaction with agent
        """
        assert isinstance(num_actions,int) and num_actions >= 2
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.num_runs = num_runs
        self.seed = seed
        self.verbose = verbose

    def predict(self, probabilities : NDArray[np.float64]) -> None:
        """
        Let the predictor set up the environment. The predictor has access to the probability distribution from which
        the agent samples its actions, but not the action itself. The predictor may adjust rewards or otherwise modify
        the environment based on this distribution.

        Arguments:
            probabilities: Probability distribution for actions by the agent
        """
        pass

    @abstractmethod
    def interact(self, action : int) -> float:
        """
        Perform the interaction of the agent with the environment, based on the action chosen by the agent.
        The interaction is purely classical, i.e. it does not depend on the agent's policy. Potential policy-dependence
        arises when the predictor sets up the environment prior to the interaction.
        hand

        Arguments:
            action: Action chosen by the agent
        
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
        self.seed += 1
        self.random = np.random.default_rng(seed = self.seed)


class BanditEnvironment(BaseEnvironment):
    """
    Multi-armed bandit environment

    Each action is associated with a fixed average reward. Upon taking an action, the reward is sampled randomly
    according to a normal distribution centred on the average value.
    Upon initialisation, the average rewards are sampled from a standard normal distribution.
    """
    def interact(self, action : int) -> float:
        return self.random.normal(self.rewards[action], 1)

    def get_optimal_reward(self) -> int:
        return self.rewards.max()

    def reset(self):
        super().reset()
        self.rewards = self.random.standard_normal((self.num_actions,))


class NewcombLikeEnvironment(BaseEnvironment):
    """
    Base class for Newcomb-like environments

    The predictor samples an action from the agent's policy.
    For predicted action i and selected action j, the reward will be reward_table[i,j]

    Arguments:
        reward_table
    """
    def __init__(self, *args,
            reward_table : list[list[float]],
            **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_actions == 2  # technical limitation for now
        assert self.num_actions == len(reward_table)
        self.reward_table = np.array(reward_table)

    def predict(self, probabilities : NDArray[np.float64]):
        prediction = utils.sample_action(self.random, probabilities)
        self.rewards = self.reward_table[prediction,:]

    def interact(self, action : int) -> float:
        return self.rewards[action]

    def get_optimal_reward(self) -> int:
        # Compute the optimal reward, based on the full reward table
        # The reward is a quadratic function of the probability of taking action 0.
        # Thus, there are three policies that could potentially be optimal
        (a,b),(c,d) = self.reward_table.tolist()
        return max(
            a,  # always take action 0
            d,  # always take action 1
            (a*d-(b+c)**2/4)/(a+d-b-c) if (a+d-b-c) < 0 else float("-inf")
                # take action 0 with probability (b+c-2*d)/(b+c-a-d)/2
        )


class NewcombEnvironment(NewcombLikeEnvironment):
    def __init__(self, *args,
        boxA : float =  5,  # guaranteed content of first box
        boxB : float = 10,  # conditional content of second box
        **kwargs):
        super().__init__(*args, reward_table=[
            [boxB, boxB+boxA],
            [0,    boxA     ]
        ], **kwargs)

class DeathInDamascusEnvironment(NewcombLikeEnvironment):
    def __init__(self, *args,
        death : float =  0,  # reward upon death
        life  : float = 10,  # reward upon survival
        **kwargs):
        super().__init__(*args, reward_table=[
            [death, life ],
            [life,  death],
        ], **kwargs)

class AsymmetricDeathInDamascusEnvironment(NewcombLikeEnvironment):
    def __init__(self, *args,
            death_in_damascus : float =  0,  # reward upon death in Damascus
            death_in_aleppo   : float =  5,  # reward upon death in Aleppo
            life              : float = 10,  # reward upon survival
            **kwargs):
        super().__init__(*args, reward_table=[
            [death_in_damascus, life           ],
            [life,              death_in_aleppo],
        ], **kwargs)

class CoordinationGameEnvironment(NewcombLikeEnvironment):
    def __init__(self, *args,
            rewardA : float = 2,  # reward in first equilibrium
            rewardB : float = 1,  # reward in second equilibrium
            **kwargs):
        super().__init__(*args, reward_table=[
            [rewardA, 0      ],
            [0,       rewardB],
        ], **kwargs)


class PolicyDependentBanditEnvironment(NewcombLikeEnvironment):
    """
    Like a multi-armed bandit, but the reward depends on the (prediction,action) pair, rather than just the action.
    This is a generalisation of both the multi-armed bandit and Newcomb-like problems
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reward_table=[[0,0],[0,0]], **kwargs)

    def reset(self):
        super().reset()
        self.reward_table = self.random.standard_normal((self.num_actions,self.num_actions))  # 2D array


class SwitchingAdversaryEnvironment(BaseEnvironment):
    """
    Classical bandit, where only one arm leads to a non-zero reward
    After a fixed number of interactions, the reward moves to a different arm
    """
    def __init__(self, *args,
            switch_at : int = None,
            **kwargs):
        super().__init__(*args, **kwargs)
        if switch_at is None:
            if self.num_steps is None:
                raise RuntimeError("SwitchingAdversaryEnvironment: require either switch_at or num_steps argument")
            switch_at = self.num_steps // 2
        self.switch_at = switch_at

    def interact(self, action : int) -> float:
        self.step += 1

        # At switch_at, the 'best' arm moves to the other side
        if self.step == self.switch_at:
            self.values = np.zeros((self.num_actions,))
            self.values[-1] = 1.0 # Move reward to the last arm

        return self.random.normal(self.values[action], 0.1)

    def get_optimal_reward(self) -> int:
        return 1.0 # The maximum reward is always 1.0

    def reset(self):
        super().reset()
        self.step = 0
        # Ensure Arm 0 is the best at the start
        self.values = np.zeros((self.num_actions,))
        self.values[0] = 1.0

