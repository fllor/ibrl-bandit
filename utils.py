import numpy as np
from numpy.typing import NDArray


def sample_action(rng : np.random.Generator, policy: NDArray[np.float64]) -> int:
    """
    Sample an action from a given policy

    Arguments:
        policy: Probability distribution over actions

    Returns:
        index of action
    """
    policy /= policy.sum()  # for numerics
    return rng.choice(len(policy), p=policy)
