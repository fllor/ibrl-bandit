import numpy as np

def simulate(env_factory, agent_factory, steps=500, runs=100):
    regret_acc = np.zeros((runs, steps))
    action_acc = np.zeros((runs, steps))
    
    for r in range(runs):
        env = env_factory()
        agent = agent_factory(k=env.num_arms)
        opt_r = env.get_best_reward()
        
        for s in range(steps):
            pred = agent.get_greedy_action()
            act = agent.get_action()
            
            # Polymorphic interact call
            reward = env.interact(act, pred)
            agent.update(act, reward, pred)
            regret_acc[r, s] = opt_r - reward
            action_acc[r, s] = act
            
    return np.cumsum(regret_acc, axis=1).mean(axis=0), action_acc
