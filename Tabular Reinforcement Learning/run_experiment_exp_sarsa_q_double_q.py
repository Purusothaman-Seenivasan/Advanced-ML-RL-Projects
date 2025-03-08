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

def plot_heatmap_frozenlake(Q, title="Frozen Lake Q-values Expected Sarsa"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(Q, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=["Left", "Down", "Right", "Up"], yticklabels=[f"State {i}" for i in range(Q.shape[0])])
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
num_episodes = int(3 * 10**4)
num_runs = 5
final_rewards = np.zeros((num_runs, num_episodes))

agent = agentfile.Agent(state_dim, action_dim)
for episode in range(num_episodes):
    reward = 0
    done = False
    agent.state = observation
    while not done: 
        #env.render()
        agent.action = agent.act(observation)
        observation, reward, done, truncated, info = env.step(agent.action)
        if reward > 0:
            reward = reward
        agent.observe(observation, reward, done)
        agent.state = observation
        if done:
            observation, info = env.reset()
    if episode % 1000 == 0:
        print(episode)

plot_heatmap_frozenlake(agent.q_table, title='Frozen Lake Q-values Q learning')

""" np.save('Expected_Sarsa_data_2.npy', final_rewards)
final_rewards = np.reshape(final_rewards, (num_runs, int(len(final_rewards[0]) / 100), 100))
final_rewards = np.mean(final_rewards, axis=2)
final_rewards_std = np.std(final_rewards, axis=0)
final_rewards = np.mean(final_rewards, axis=0)
np.save('Expected_Sarsa_average_2.npy', final_rewards)
np.save('Expected_Sarsa_std_2.npy', final_rewards_std)
plt.xlabel('Epochs')
plt.ylabel('Average Score over 100 Epochs')
plt.plot(final_rewards)
plt.fill_between(range(len(final_rewards)), final_rewards - 2 * final_rewards_std, final_rewards + 2 * final_rewards_std, color='r', alpha=0.2)
plt.show()
plt.savefig('Expected_Sarsa_Average_2') """