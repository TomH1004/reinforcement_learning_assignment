import glob
import logging, logging.config
import os
import re

from matplotlib import pyplot as plt
import logger
import numpy as np
import plotly.io as pio

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
    ax.set_xlabel('Episodes', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'Moving Reward avg. (Window Size = {window_size})', fontsize=16, fontweight='bold')
    ax.set_title(f'Compare agents ({info})',  fontsize=16, fontweight='bold')
    ax.legend(loc='center right', fontsize=14)

    # Anpassungen für die Ticks
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    # Anzeigen des Plots
    plt.show()

def pltRewards(expFol="zE01", window_size=50):
    fl_rewards = []
    agentNames = []
    expFolder = expFol
    pathsReward353 = glob.glob(os.path.join('share', 'LAST', expFolder, '**', '*-*-reward_sums*.npy'), recursive=True)
    agentPattern353 = re.compile(r'.*-.*-reward_sums_([A-Za-z\s-]+)_.*')
    for path in pathsReward353:
        sums=np.load(path, allow_pickle=True)
        agentNames.append(agentPattern353.search(path).group(1))
        fl_rewards.append(np.convolve(sums, np.ones(window_size)/window_size, mode='valid'))    
    if len(fl_rewards)>0:
        fl_rewards_trimmed = [rewards[:len(fl_rewards[0])] for rewards in fl_rewards]
        x = range(len(fl_rewards[0]))
        y = fl_rewards_trimmed
        plotAgentRewards(x,y,agentNames, window_size, "575 race3")

def plotOptimization():
    optFolder = "zOpt03"
    optimized_params = glob.glob(os.path.join('model_storage', optFolder, '**', '*-575-optimized*.json'))
    optimized_params_np = glob.glob(os.path.join('model_storage', optFolder, '**', '*-575-optimized*.npy'))
    statePattern = re.compile(r'.*-(\d{3})-optimized_params_([A-Za-z\s-]+)_.*\.json')
    optimizations = []
    # just to read 
    for path in optimized_params_np:
        optimizations.append(np.load(path, allow_pickle=True))
        optimizations
    for path in optimized_params:
        stateName = statePattern.search(path).group(1)
        agentName = statePattern.search(path).group(2)
        with open(path, 'r') as f:
            fig_json = f.read()
        loaded = pio.from_json(fig_json)
        loaded.update_layout(
            title={"text": f"{agentName} - Race 3 - 500 episodes - {stateName}", "font": {"size": 30}},
            xaxis={"title": {"font": {"size": 30, "weight": "bold"}}},
            yaxis={"title": {"font": {"size": 30, "weight": "bold"}}},
            font={"size": 25, "weight": "bold"}
        )
        loaded.show()
        f.close()

def plot_goal_rates():
    agentName = ''
    subFolder = "zE04"
    map_goal_paths = glob.glob(os.path.join('share', 'LAST', subFolder, '**', '*-map_goal_rates*.npz'))
    statePattern = re.compile(r'.*-map_goal_rates_([A-Za-z\s-]+)_.*\.npz')    
    for path in map_goal_paths:
        agentName = statePattern.search(path).group(1)
        map_goal_and_total = np.load(path, allow_pickle=True)
        map_rewards_goal_rates = map_goal_and_total['arr_0'].item()
        total = map_goal_and_total['arr_1'].item()
        map_names = list(map_rewards_goal_rates.keys())
        goal_rates = [data['goal_rate'] for data in map_rewards_goal_rates.values()]
        
        # Add overall goal rate to the data
        map_names.append('Overall')
        goal_rates.append(total)
        
        plt.figure(figsize=(12, 6))
        plt.bar(map_names, goal_rates, color='blue')
        plt.xlabel('Map')
        plt.ylabel('Goal Rate (%)')
        plt.title(f'{agentName} - Goal Rate per Map and Overall')
        plt.xticks(rotation=45)
        plt.grid(True)
    #    plt.savefig()
        plt.show()


if __name__ == "__main__":
    #opt_sarsa = np.load("model_storage/2024-06-01_10-33-05/world3optimized-params_SARSA_2024-06-01_10-33-05.npy", allow_pickle=True)
    #opt_q = np.load("model_storage/2024-06-01_10-33-13/world3optimized-params_Q-Learning_2024-06-01_10-33-13.npy", allow_pickle=True)

    plot_goal_rates()

    #plotOptimization()
    #pltRewards("zE02", 175)
    #pltRewards("zE03", 200)
    #pltRewards("zE04", 20)