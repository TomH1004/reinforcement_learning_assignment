import glob
import logging, logging.config
import os
import re

from matplotlib import pyplot as plt
import logger
import numpy as np
import plotly.io as pio

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
    color_dict = {
        'SARSA': "b",
        'Q-Learning': "r",
        'Double Q-Learning': "g"
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, rewards in enumerate(y):
        ax.plot(x, rewards, color=color_dict[agentNames[idx]], label=agentNames[idx])
    

    #ax.plot(x, y[1], color=color_dict[agentNames[1]], label=agentNames[1])


    #ax.plot(x, y[2], color=color_dict[agentNames[2]], label=agentNames[2])

    # Hinzufügen von Legende und Labels
    ax.set_xlabel('Episodes')
    ax.set_ylabel(f'Moving Reward avg. (Window Size = {window_size})')
    ax.set_title(f'Compare agents ({info})')
    ax.legend(loc='upper right')

    # Anzeigen des Plots
    plt.show()

def pltRewards():
    window_size = 200
    fl_rewards = []
    agentNames = []
    expFolder = "zE03"
    pathsReward353 = glob.glob(os.path.join('model_storage', expFolder, '**', '*-353-reward_sums*'), recursive=True)
    agentPattern353 = re.compile(r'.*-353-reward_sums_([A-Za-z\s-]+)_.*')
    for path in pathsReward353:
        sums=np.load(path, allow_pickle=True)
        agentNames.append(agentPattern353.search(path).group(1))
        fl_rewards.append(np.convolve(sums, np.ones(window_size)/window_size, mode='same'))    
    if len(fl_rewards)>0:
        x = range(len(fl_rewards[0]))
        y = fl_rewards
        plotAgentRewards(x,y,agentNames, window_size, "353 race3")
    

    fl_rewards = []
    agentNames = []
    pathsReward474 = glob.glob(os.path.join('model_storage', expFolder, '**', '*-474-reward_sums*'), recursive=True)
    agentPattern474 = re.compile(r'.*-474-reward_sums_([A-Za-z\s-]+)_.*')
    for path in pathsReward474:
        sums=np.load(path, allow_pickle=True)
        agentNames.append(agentPattern474.search(path).group(1))
        fl_rewards.append(np.convolve(sums, np.ones(window_size)/window_size, mode='same'))
    if len(fl_rewards)>0:
        x = range(len(fl_rewards[0]))
        y = fl_rewards
        plotAgentRewards(x,y,agentNames, window_size, "474 race3")

    fl_rewards = []
    agentNames = []
    pathsReward575 = glob.glob(os.path.join('model_storage', expFolder, '**', '*-575-reward_sums*'), recursive=True)
    agentPattern575 = re.compile(r'.*-575-reward_sums_([A-Za-z\s-]+)_.*')
    for path in pathsReward575:
        sums=np.load(path, allow_pickle=True)
        agentNames.append(agentPattern575.search(path).group(1))
        fl_rewards.append(np.convolve(sums, np.ones(window_size)/window_size, mode='same'))    
    if len(fl_rewards)>0:
        x = range(len(fl_rewards[0]))
        y = fl_rewards
        plotAgentRewards(x,y,agentNames, window_size, "575 race3")

def plotOptimization():
    optFolder = "zOpt01"
    optimized_params = glob.glob(os.path.join('model_storage', optFolder, '**', '*-575-optimized*.json'))
    statePattern = re.compile(r'.*-(\d{3})-optimized_params_([A-Za-z\s-]+)_.*\.json')
    for path in optimized_params:
        stateName = statePattern.search(path).group(1)
        agentName = statePattern.search(path).group(2)
        with open(path, 'r') as f:
            fig_json = f.read()
        loaded = pio.from_json(fig_json)
        loaded.update_layout(
            title={"text": f"{agentName} - Race 3 - 500 episodes - {stateName}", "font": {"size": 20}},
            xaxis={"title": {"font": {"size": 18, "weight": "bold"}}},
            yaxis={"title": {"font": {"size": 18, "weight": "bold"}}}, 
            font={"size": 17, "weight": "bold"}  # Default font size for other text elements
        )
        loaded.show()
        f.close()

if __name__ == "__main__":
    #opt_sarsa = np.load("model_storage/2024-06-01_10-33-05/world3optimized-params_SARSA_2024-06-01_10-33-05.npy", allow_pickle=True)
    #opt_q = np.load("model_storage/2024-06-01_10-33-13/world3optimized-params_Q-Learning_2024-06-01_10-33-13.npy", allow_pickle=True)
    optFolder = "zOpt03"
    optimized_params = glob.glob(os.path.join('model_storage', optFolder, '**', '*-575-optimized*.npy'))
    opt_dq = np.load(optimized_params[0], allow_pickle=True)

    #plotOptimization()
    pltRewards()
