import argparse
import numpy as np
import agents
import environments
import utils

def parse_argument_string(string : str) -> tuple[str, dict[str, float]]:
    """
    Parse a string that optionally includes arguments.

    Input syntax:
        No arguments:       <string>
        Single argument:    <string>:<arg1>=<val1>
        Multiple arguments: <string>:<arg1>=<val1>,<arg2>=<val2>,...

    All argument values should either be floats, or tuples of floats separated by ":"

    Returns a tuple consisting of:
        the base string
        a dict of all options

    Example:
        string = "base:opt1=1,opt2=2:0.5"
    ->  "base",{"opt1":1.,"opt2":(2,0.5)}
    """
    if ":" not in string:
        return string, dict()

    name,args_str = string.split(":",1)
    args_dict = dict()
    for arg in args_str.split(","):
        arg_name,arg_val = arg.split("=",1)
        if ":" not in arg_val:
            args_dict[arg_name] = float(arg_val)
        else:
            args_dict[arg_name] = tuple(map(float, arg_val.split(":")))
    return name, args_dict


def construct_agent(string : str, options : dict[str,int]) -> agents.BaseAgent:
    """
    Construct agent from a given string, possibly including agent-specific options
    """

    agent_types = {
        "classical":        agents.QLearningAgent,
        "bayesian":         agents.BayesianAgent,
        "exp3":             agents.EXP3Agent,
        "experimental1":    agents.ExperimentalAgent1,
        "experimental2":    agents.ExperimentalAgent2
    }

    name, kwargs = parse_argument_string(string)
    if name not in agent_types:
        raise RuntimeError("Invalid agent type: " + name)

    arguments = dict()
    arguments.update(options)
    arguments.update(kwargs)
    arguments.pop("num_steps")
    arguments.pop("num_runs")
    return agent_types[name](**arguments)


def construct_environment(string : str, options : dict[str,int]) -> environments.BaseEnvironment:
    """
    Construct environment from a given string, possibly including environment-specific options
    """

    environment_types = {
        "bandit":               environments.BanditEnvironment,
        "switching":            environments.SwitchingAdversaryEnvironment,
        "newcomb":              environments.NewcombEnvironment,
        "damascus":             environments.DeathInDamascusEnvironment,
        "asymmetric-damascus":  environments.AsymmetricDeathInDamascusEnvironment,
        "coordination":         environments.CoordinationGameEnvironment,
        "pdbandit":             environments.PolicyDependentBanditEnvironment,
    }

    name, kwargs = parse_argument_string(string)
    if name not in environment_types:
        raise RuntimeError("Invalid environment: " + options.environment)

    arguments = dict()
    arguments.update(options)
    arguments.update(kwargs)
    return environment_types[name](**arguments)


def simulate(
        env : environments.BaseEnvironment,
        agent : agents.BaseAgent,
        options : dict):
    base_rng = np.random.default_rng(options["seed"])
    num_steps = options["num_steps"]
    num_runs = options["num_runs"]
    verbose = options["verbose"]

    average_reward = np.zeros((num_steps,))
    average_reward_sq = np.zeros((num_steps,))
    optimal_reward = 0
    for r in range(num_runs):
        #if verbose > 0:
        #    print(f"Run {r+1}/{num_runs}")
        env.reset()
        agent.reset()
        optimal_reward += env.get_optimal_reward()
        for i in range(num_steps):
            policy = agent.get_policy()
            action = utils.sample_action(base_rng, policy)
            reward = env.interact(action, policy)
            #if verbose > 0:
            #    print(i, action, reward, policy.tolist(), agent.q.tolist())
            agent.update(policy, action, reward)
            average_reward[i] += reward
            average_reward_sq[i] += reward**2

    average_reward /= num_runs
    average_reward_sq /= num_runs
    average_reward_spread = np.sqrt(average_reward_sq - average_reward**2)
    average_reward_unc = average_reward_spread / np.sqrt(num_runs)
    optimal_reward /= num_runs

    for i in range(num_steps):
        print(
            i,
            optimal_reward,             # Expected reward when using optimal policy (as determined by environment)
            average_reward[i],          # Average reward received by agent
            average_reward_unc[i],      # Uncertainty of reward (converges to 0 as num_runs goes to infinity)
            average_reward_spread[i],   # Spread of reward (converges to constant)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL test with multi-armed bandit")
    parser.add_argument("environment",          help="Environment type",            type=str)
    parser.add_argument("agent",                help="Agent type",                  type=str)
    parser.add_argument("-k", "--arms",         help="Number of arms",              type=int,       default=2)
    parser.add_argument("-s", "--steps",        help="Number of steps per episode", type=int,       default=1001)
    parser.add_argument("-r", "--runs",         help="Number of episodes to run",   type=int,       default=1)
    parser.add_argument(      "--seed",         help="Random number seed",          type=int,       default=42)
    parser.add_argument("-v", "--verbose",      help="Debug output",                action="count", default=0)
    parsed_args = parser.parse_args()
    options = {
        "num_actions": parsed_args.arms,
        "num_steps":   parsed_args.steps,
        "num_runs":    parsed_args.runs,
        "seed":        parsed_args.seed,
        "verbose":     parsed_args.verbose
    }
    env = construct_environment(parsed_args.environment, options)
    agent = construct_agent(parsed_args.agent, options)
    simulate(env, agent, options)
