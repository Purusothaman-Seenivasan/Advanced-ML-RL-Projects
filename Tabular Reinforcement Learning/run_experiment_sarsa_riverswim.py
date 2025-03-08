import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

def plot_heatmap_riverswim(Q, title="RiverSwim Q-values Expected Sarsa"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(Q, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=["Left", "Right"], yticklabels=[f"State {i}" for i in range(Q.shape[0])])
    plt.title(title)
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.savefig(title)
    plt.show()

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

observation, info = env.reset()
num_episodes = 1
num_steps = (3 * 10**4)
num_runs = 1
final_rewards = np.zeros((num_runs, num_steps))


agent = agentfile.Agent(state_dim, action_dim)
for episode in range(num_episodes):
    done = False
    agent.state = observation
    agent.action = agent.act(observation)
    for j in range(num_steps): 
        #env.render()
        observation, reward, done, truncated, info = env.step(agent.action)
        agent.observe(observation, reward, done)
        agent.state = observation
        if j % 1000 == 0:
            print(j)
plot_heatmap_riverswim(agent.q_table, title='RiverSwim Q-values Sarsa')


""" 
np.save('Sarsa_data_river.npy', final_rewards)
final_rewards = np.reshape(final_rewards, (num_runs, int(len(final_rewards[0]) / 100), 100))
final_rewards = np.mean(final_rewards, axis=2)
final_rewards_std = np.std(final_rewards, axis=0)
final_rewards = np.mean(final_rewards, axis=0)
np.save('Sarsa_average_river.npy', final_rewards)
np.save('Sarsa_std_river.npy', final_rewards_std)
plt.xlabel('Epochs')
plt.ylabel('Average Score over 100 Epochs')
plt.plot(final_rewards)
plt.fill_between(range(len(final_rewards)), final_rewards - 2 * final_rewards_std, final_rewards + 2 * final_rewards_std, color='r', alpha=0.2)
plt.show()
plt.savefig('Sarsa_Average_river') """