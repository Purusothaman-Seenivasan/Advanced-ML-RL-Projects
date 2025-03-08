import numpy as np 
import matplotlib.pyplot as plt

Q_learning = np.load('Q_learning_data_river.npy')
double_Q_learning = np.load('Double_Q_learning_data_river.npy')
expected_sarsa = np.load('Expected_Sarsa_data_river.npy')
sarsa = np.load('Sarsa_data_river.npy')

Q_learning = np.reshape(Q_learning, (5, int(len(Q_learning[0]) / 100), 100))
Q_learning = np.mean(Q_learning, axis=2)
Q_learning_std = np.std(Q_learning, axis=0)
Q_learning = np.mean(Q_learning, axis=0)

double_Q_learning = np.reshape(double_Q_learning, (5, int(len(double_Q_learning[0]) / 100), 100))
double_Q_learning = np.mean(double_Q_learning, axis=2)
double_Q_learning_std = np.std(double_Q_learning, axis=0)
double_Q_learning = np.mean(double_Q_learning, axis=0)

expected_sarsa = np.reshape(expected_sarsa, (5, int(len(expected_sarsa[0]) / 100), 100))
expected_sarsa = np.mean(expected_sarsa, axis=2)
expected_sarsa_std = np.std(expected_sarsa, axis=0)
expected_sarsa = np.mean(expected_sarsa, axis=0)

sarsa = np.reshape(sarsa, (5, int(len(sarsa[0]) / 100), 100))
sarsa = np.mean(sarsa, axis=2)
sarsa_std = np.std(sarsa, axis=0)
sarsa = np.mean(sarsa, axis=0)

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Average Score over 100 Epochs')

plt.plot(Q_learning[:125], color='purple', label='Q-learning')


plt.plot(double_Q_learning[:125], color='blue', label='Double Q-learning')

plt.plot(expected_sarsa[:125], color='red', label='Expected Sarsa')

plt.plot(sarsa[:125], color='green', label='Sarsa')

plt.legend()


plt.savefig('Comparison_river_shortened')

plt.show()

print()