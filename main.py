import argparse
import numpy as np
import agents
import environments
import utils


def main(options):
    # Quietly skip this combination (experimental agent 2 does not accept softmax policy)
    if options.agent.startswith("experimental2") and options.policy.startswith("softmax"):
        return

    np.random.seed(options.seed)
    num_steps = options.steps
    num_runs = options.runs
    if options.environment == "bandit":
        env = environments.BanditEnvironment(options.arms)
    elif options.environment == "newcomb":
        assert options.arms == 2
        env = environments.NewcombEnvironment()
    elif options.environment == "damascus":
        assert options.arms == 2
        env = environments.DeathInDamascusEnvironment()
    elif options.environment == "asymmetric-damascus":
        assert options.arms == 2
        env = environments.AsymmetricDeathInDamascusEnvironment()
    elif options.environment == "coordination":
        assert options.arms == 2
        env = environments.CoordinationGameEnvironment()
    elif options.environment == "pdbandit":
        assert options.arms == 2
        env = environments.PolicyDependentBanditEnvironment()
    else:
        raise RuntimeError("Invalid environment: " + options.environment)

    if options.policy.startswith("epsilon"):
        policy_function = agents.epsilon_greedy
    elif options.policy.startswith("softmax"):
        policy_function = agents.softmax
    else:
        raise RuntimeError("Invalid policy type: " + options.agent)

    if options.agent.startswith("classical"):
        agent = agents.QLearningAgent(options.arms, policy_function, options.learning_rate)
    elif options.agent.startswith("bayesian"):
        agent = agents.BayesianAgent(options.arms, policy_function)
    elif options.agent.startswith("experimental1"):
        agent = agents.ExperimentalAgent1(options.arms, policy_function, options.learning_rate)
    elif options.agent.startswith("experimental2"):
        agent = agents.ExperimentalAgent2(options.arms, options.learning_rate)
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
            action = utils.sample_action(policy)
            reward = env.interact(action, policy)
            if options.verbose > 0:
                print(i, action, reward, policy.tolist(), agent.q.tolist())
            agent.update(policy, action, reward)
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
