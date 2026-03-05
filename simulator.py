import numpy as np
import agents
import environments
import utils


def simulate(
        env : environments.BaseEnvironment,
        agent : agents.BaseAgent,
        options : dict) -> dict:
    """
    Simulate interactions between agent and environment

    Arguments:
        env:     the environment
        agent:   the agent
        options: dictionary of further options, namely
            num_steps: Number of steps to simulate in each run
            num_runs:  Number of runs
            verbose:   Request debugging output

    Returns:
        A dictionary containing summary information, namely
            average_reward: a (2,num_steps) array containing the average reward and squared reward^2 at each step
            optimal_reward: the expected reward for an agent with optimal policy
    """
    num_steps = options.get("num_steps", 101)
    num_runs = options.get("num_runs", 1)
    verbose = options.get("verbose", 0)

    average_reward = np.zeros((2,num_steps)) # average reward, average (reward^2)
    optimal_reward = 0

    for r in range(num_runs):
        env.reset()
        agent.reset()
        optimal_reward += env.get_optimal_reward()
        for i in range(num_steps):
            probabilities = agent.get_probabilities()
            env.predict(probabilities)
            action = utils.sample_action(agent.random, probabilities)
            reward = env.interact(action)
            agent.update(probabilities, action, reward)
            average_reward[0,i] += reward
            average_reward[1,i] += reward**2

    average_reward /= num_runs
    optimal_reward /= num_runs

    results = {
        "average_reward": average_reward,
        "optimal_reward": optimal_reward
    }

    return results