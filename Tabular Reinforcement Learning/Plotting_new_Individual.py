import numpy as np 
import matplotlib.pyplot as plt

Q_learning = np.load('Q_learning_data_2.npy')
double_Q_learning = np.load('Double_Q_learning_data_2.npy')
expected_sarsa = np.load('Expected_Sarsa_data_1.npy')
sarsa = np.load('Sarsa_data_2.npy')

#Q_learning = np.reshape(Q_learning, (5, int(len(Q_learning[0]) / 100), 100))
Q_learning_sum  = np.cumsum(Q_learning, axis=1)
Q_learning_sum /= np.arange(1, len(Q_learning[0]) + 1)
Q_learning_std = np.std(Q_learning_sum, axis=0)
Q_learning_final = np.mean(Q_learning_sum, axis=0)
bottom, top = Q_learning_final - 1.96 * Q_learning_std, Q_learning_final + 1.96 * Q_learning_std

#double_Q_learning = np.reshape(double_Q_learning, (5, int(len(double_Q_learning[0]) / 100), 100))
double_Q_learning_sum = np.cumsum(double_Q_learning, axis=1)
double_Q_learning_sum /= np.arange(1, len(Q_learning[0]) + 1)
double_Q_learning_std = np.std(double_Q_learning_sum, axis=0)
double_Q_learning_final = np.mean(double_Q_learning_sum, axis=0)

#expected_sarsa = np.reshape(expected_sarsa, (5, int(len(expected_sarsa[0]) / 100), 100))
expected_sarsa_sum = np.cumsum(expected_sarsa, axis=1)
expected_sarsa_sum /= np.arange(1, len(Q_learning[0]) + 1)
expected_sarsa_std = np.std(expected_sarsa_sum, axis=0)
expected_sarsa_final = np.mean(expected_sarsa_sum, axis=0)

#sarsa = np.reshape(sarsa, (5, int(len(sarsa[0]) / 100), 100))
sarsa_sum = np.cumsum(sarsa, axis=1)
sarsa_sum /= np.arange(1, len(Q_learning[0]) + 1)
sarsa_std = np.std(sarsa_sum, axis=0)
sarsa_final = np.mean(sarsa_sum, axis=0)

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Cumulative Average')
plt.plot(Q_learning_final, color='purple', label='Q-learning')
plt.fill_between(np.arange(20000), Q_learning_final - 1.96 * Q_learning_std, Q_learning_final + 1.96 * Q_learning_std, color= 'purple', alpha=0.2)
plt.title('Q-learning on Frozen Lake')
plt.savefig('Q-learning on Frozen Lake')

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Cumulative Average')
plt.plot(double_Q_learning_final, color='blue', label='Double Q-learning')
plt.fill_between(np.arange(20000), double_Q_learning_final - 1.96 * double_Q_learning_std, double_Q_learning_final + 1.96 * double_Q_learning_std, color= 'blue', alpha=0.2)
plt.title('Double Q-learning on Frozen Lake')
plt.savefig('Double Q-learning on Frozen Lake')

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Cumulative Average')
plt.plot(expected_sarsa_final, color='red', label='Expected Sarsa')
plt.fill_between(np.arange(20000), expected_sarsa_final - 1.96 * expected_sarsa_std, expected_sarsa_final + 1.96 * expected_sarsa_std, color= 'red', alpha=0.2)
plt.title('Expected Sarsa on Frozen Lake')
plt.savefig('Expected Sarsa on Frozen Lake')

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Cumulative Average')
plt.plot(sarsa_final, color='green', label='Sarsa')
plt.fill_between(np.arange(20000), sarsa_final - 1.96 * sarsa_std, sarsa_final + 1.96 * sarsa_std, color= 'green', alpha=0.2)
plt.title('Sarsa on Frozen Lake')
plt.savefig('Sarsa on Frozen Lake')
plt.legend()