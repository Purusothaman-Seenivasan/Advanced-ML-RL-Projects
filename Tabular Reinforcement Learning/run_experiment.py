import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)




try:
    env = gym.make(args.env, render_mode='rgb_array')
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)


action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)

observation, info = env.reset()
num_episodes = int(2 * 10**4)
final_rewards = np.zeros(num_episodes)

for episode in range(num_episodes):
    agent.previous_state = observation
    reward = 0
    done = False
    while not done: 
        #env.render()
        action = agent.act(observation) 
        observation, reward, done, truncated, info = env.step(action)
        if reward > 0:
            reward = reward
        agent.observe(observation, reward, done)
        agent.previous_state = observation
        if done:
            final_rewards[episode] = reward
            observation, info = env.reset()
    if episode % 1000 == 0:
        print(episode)
env.close()


cumulative_rewards = np.cumsum(final_rewards)
plt.plot(cumulative_rewards)
plt.show()
final_rewards = np.reshape(final_rewards, (int(final_rewards / 1000), 1000))
final_rewards = np.mean(final_rewards, axis=1)
print(final_rewards)
print()