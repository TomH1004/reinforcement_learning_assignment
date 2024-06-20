import glob
import logging, logging.config
import os
import re

from matplotlib import pyplot as plt
import logger
import numpy as np

class DiscretizeHelper:
    
    def __init__(self, logger):
        self.logger=logger

    def preprocessing_observations(self, observation, state_len):
        # states_list = [['west'], ['north'], ['east']]
        __car_X = 50
        __car_size = 24  # cm
        __resize_factor = __car_X / __car_size
        __obs_discrete = []

        if observation < (5 + (__car_size * __resize_factor)):
            # obstacle detected
            __value = 0
        elif observation < (10 * __resize_factor + __car_size):
            # close obstacle detected
            __value = 1
        elif observation < (15 * __resize_factor + __car_size):
            # obstacle detected
            __value = 2
        elif observation < (20 * __resize_factor + (__car_size)):
            # obstacle detected
            __value = 3
        elif observation < (25 * __resize_factor + (__car_size)):
            # obstacle detected
            __value = 4
        elif observation < (30 * __resize_factor + (__car_size)):
            # obstacle detected
            __value = 5
        elif observation < (35 * __resize_factor + (__car_size)):
            # obstacle detected
            __value = 6
        elif observation < (40 * __resize_factor + (__car_size)):
            # obstacle detected
            __value = 7
        else:
            # no obstacle detected
            __value = 8
        return min(state_len-1, __value)


    def preprocess(self, observations, states_list):
        __obs_discrete = []
        for idx, state in enumerate(states_list):
            value = self.preprocessing_observations(observations[idx], len(state))
            __obs_discrete.append(value)
        return __obs_discrete
    

def plotAgentRewards(x,y,agentNames, window_size, info=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot für SARSA
    ax.plot(x, y[0], color=color_dict[agentNames[0]], label=agentNames[0])
    
    # Plot für Q-Learning
    ax.plot(x, y[1], color=color_dict[agentNames[1]], label=agentNames[1])

    # Plot für Double Q-Learning
    ax.plot(x, y[2], color=color_dict[agentNames[2]], label=agentNames[2])

    # Hinzufügen von Legende und Labels
    ax.set_xlabel('Episodes')
    ax.set_ylabel(f'Moving Reward avg. (Window Size = {window_size})')
    ax.set_title(f'Compare agents ({info})')
    ax.legend()

    # Anzeigen des Plots
    plt.show()

if __name__ == "__main__":
    #opt_sarsa = np.load("model_storage/2024-06-01_10-33-05/world3optimized-params_SARSA_2024-06-01_10-33-05.npy", allow_pickle=True)
    #opt_q = np.load("model_storage/2024-06-01_10-33-13/world3optimized-params_Q-Learning_2024-06-01_10-33-13.npy", allow_pickle=True)
    #opt_dq = np.load("model_storage/2024-06-01_10-33-24/world3optimized-params_Double Q-Learning_2024-06-01_10-33-24.npy", allow_pickle=True)
    window_size = 200
    color_dict = {
        'SARSA': "b",
        'Q-Learning': "r",
        'Double Q-Learning': "g"
    }
    fl_rewards = []
    agentNames = []

    pathsReward353 = glob.glob(os.path.join('share', 'zE01', '**', '*-353-reward_sums*'), recursive=True)
    agentPattern353 = re.compile(r'.*-353-reward_sums_([A-Za-z\s-]+)_.*')
    for path in pathsReward353:
        sums=np.load(path, allow_pickle=True)
        agentNames.append(agentPattern353.search(path).group(1))
        fl_rewards.append(np.convolve(sums, np.ones(window_size)/window_size, mode='same'))    
    x = range(len(fl_rewards[0]))
    y = fl_rewards
    plotAgentRewards(x,y,agentNames, window_size, "353 race3")
    

    fl_rewards = []
    agentNames = []
    pathsReward474 = glob.glob(os.path.join('model_storage', 'zE01', '**', '*-474-reward_sums*'), recursive=True)
    agentPattern474 = re.compile(r'.*-474-reward_sums_([A-Za-z\s-]+)_.*')
    for path in pathsReward474:
        sums=np.load(path, allow_pickle=True)
        agentNames.append(agentPattern474.search(path).group(1))
        fl_rewards.append(np.convolve(sums, np.ones(window_size)/window_size, mode='same'))
    x = range(len(fl_rewards[0]))
    y = fl_rewards
    plotAgentRewards(x,y,agentNames, window_size, "474 race3")


    fl_rewards = []
    agentNames = []
    pathsReward575 = glob.glob(os.path.join('model_storage', 'zE01', '**', '*-575-reward_sums*'), recursive=True)
    agentPattern575 = re.compile(r'.*-575-reward_sums_([A-Za-z\s-]+)_.*')
    for path in pathsReward575:
        sums=np.load(path, allow_pickle=True)
        agentNames.append(agentPattern575.search(path).group(1))
        fl_rewards.append(np.convolve(sums, np.ones(window_size)/window_size, mode='same'))    
    x = range(len(fl_rewards[0]))
    y = fl_rewards
    plotAgentRewards(x,y,agentNames, window_size, "575 race3")
#    paths353 = glob.glob(os.path.join('model_storage', 'zE01', '**', '*-353-floating_rewards*'), recursive=True)
#    agentPattern353 = re.compile(r'.*353-floating_rewards_([A-Za-z\s-]+)_.*')
#    for path in paths353:
#        agentNames.append(agentPattern353.search(path).group(1))
#        fl_rewards_353.append(np.load(path, allow_pickle=True))