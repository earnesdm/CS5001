import numpy as np
import matplotlib.pyplot as plt

offline_dqn_data = np.load("results/ACORL_results/ACORL_noreg_Evaluations.npy")
ACORL_data = [0.0] + list(np.load("results/ACORL_results/ACORL_Evaluations.npy"))

final_online_score = 13.6

plt.axhline(y=final_online_score, color = 'red', label='DQN-Online', linewidth=3)
plt.plot(offline_dqn_data, color='blue', label='DQN-Offline')
plt.plot(ACORL_data, color='green', label='ACORL')
plt.legend()
plt.xlabel('50,000 Time Steps')
plt.ylabel('Reward')
plt.title('Performance of Offline RL Approaches (With Online DQN Baseline)')
plt.show()

print(ACORL_data)
