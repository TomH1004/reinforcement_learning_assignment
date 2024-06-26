from functools import partial
import sys
import os

from datetime import datetime
import logging
import logging.config
import logger
import json

import numpy as np
import gym
from reinforcement_agents.agents import TemporalDifferenceLearning as TDL
from sim_world.envs.car_0.ev3_sim_car import SimCar as Car
from sim_world.envs.pygame_0.ev3_sim_pygame_2d_V2 import PyGame2D as Simulation
import optuna
import optuna.visualization as vis
import random
import argparse
import matplotlib.pyplot as plt
from sim_world.maps_config import get_map_config, maps_config

from preprocess import DiscretizeHelper as DH

# only for lecturer otherwise comment out the following line
# path_to_main = ".\student\simulation"
path_to_main = ".\simulation"



os.chdir(os.getcwd() + path_to_main)

def load_policy(policy_as_json):
    """load the policy json file

    Args:
        policy_as_json (str): path to json file

    Returns:
        dict: policy dict: {state:action}
    
    """
    __policy = {}
    with open(policy_as_json, 'r') as __f:
      __policy = json.load(__f)
    return __policy

def run_model(env, policy, states_list):
    """run the car with the given policy

    Args:
        car (SimCar): car to drive given the policy
        policy (dict): policy
    """
    __state = env.reset()

    __done = False
    total_steps = 0
    total_reward = 0
    
    while (__done == False):
        total_steps += 1
        # preprocessing of the measured values
        __preprocessed_state = pp.preprocess(observations=__state, states_list=states_list)

        # select the best action; based on state
        __action = greedy(policy=policy, state=__preprocessed_state)
        __state, __reward, __done, __info = env.step(__action)
        total_reward += __reward
        
    logger.info('Run Statistics: Steps = %d, Reward = %.2f',
                total_steps, total_reward)
            
def greedy(policy, state):
    '''
    Greedy policy

    return the best action corresponding to the state
    '''
    __key = str(state[0]) + ' ' + str(state[1]) + ' ' + str(state[2])
    return policy[__key]


def train_model(env, agent, states_list, file_path, file_prefix, file_suffix, q_table=None):
    """train the agent in the given env

    Args:
        env (gym.Env): gym.Env to train in 
        agent (agent): reinforcement agent to train
        file_prefix (strings): #TODO
        file_suffix (strings): #TODO
        q_table (np.array, optional): q_table used for init. Defaults to None.
    """
    __observation_space_nums = __get_observation_space_num(env=env, states_list=states_list)
    if (not (q_table is None)):
        agent.actionValueTable = q_table
        logger.info('USE GIVEN Q-TABLE')

    __reward_sums, __evst, __actionValueTable_history, stats, map_rewards_goal_rates = _runExperiment_NStep(agent_nEpisodes=agent_nEpisodes, env=env, agent=agent, states_list=states_list, observation_space_num=__observation_space_nums)
    rewards_smooth = plot_rewards(__reward_sums, map_rewards_goal_rates)
    np.save(file_path + file_prefix + 'map_reward_goal_rates' + file_suffix + '.npy', map_rewards_goal_rates)
    np.save(file_path + file_prefix + 'floating_rewards' + file_suffix + '.npy', rewards_smooth)
    np.save(file_path + file_prefix + 'q-table' + file_suffix + '.npy', __actionValueTable_history[-1])
    np.save(file_path + file_prefix + 'reward_sums' + file_suffix + '.npy', __reward_sums)

    # create argmax policy; store as json
    __json_path = file_path + file_prefix + 'policy' + file_suffix + '.json'
    __q_policy = __convert_q_table_to_policy(q_table=__actionValueTable_history[-1], observation_space_num=__observation_space_nums)
    __store_dict_as_json(dict_data=__q_policy, file_name=__json_path)

    logger.info('Training Statistics: Avg Steps = %.2f, Avg Reward = %.2f, Goal Rate = %.2f%%',
                stats['avg_steps'], stats['avg_reward'], stats['goal_rate'])

def test_model(env, agent, states_list, file_path, file_prefix, file_suffix, q_table=None):
    # load rewards and q-table
    __rewards = read_numpy_data(numpy_file=file_path + file_prefix + 'reward_sums' + file_suffix)
    logger.debug('rewards loaded = \'%s\'', str(__rewards))
    __q_data = read_numpy_data(numpy_file=file_path + file_prefix + 'q-table' + file_suffix)

    # test the q-table
    map_goal_rates, total_goal_rate = _test_q_table(q_table=__q_data, env=env, states_list=states_list, agent_nEpisodes=100)
    np.savez(file_path + file_prefix + 'map_goal_rates' + file_suffix + '.npy', map_goal_rates, total_goal_rate)
    # Plot the goal rates
    #plot_goal_rates(map_goal_rates, total_goal_rate)

################################################################
### Additional functions only for the simulation environment ###
################################################################

def _runExperiment_NStep(agent_nEpisodes, env, agent, states_list, observation_space_num):
    """Train and test the agent in the given Environment for the given Episodes.
       Function for N-Step agents.

      Args:
          agent_nEpisodes (int): number of Episoeds to train
          env (gym env): Evironment to train/test the agent in
          agent (agent): the agent to train
          observation_space_num (list): number of states per dimenson of observation

      Returns:
          list: reward_sums
          list: episodesvstimesteps 
          list: actionValueTable_history
          dict: statistics
          dict: map_rewards_goal_rates
    """
    __reward_sums = []
    __episodesvstimesteps = []
    __actionValueTable_history = []
    total_steps = 0
    total_rewards = 0
    goals_reached = 0
    randomMap = False
    map_rewards_goal_rates = {}

    for __e in range(agent_nEpisodes):
        __timesteps = 0
        if (__e % 100 == 0):
            logger.info('Episode: %s', str(__e))

        if randomMap:
            config = setup_random_map()
            MAP = config["path"]
            MAP_START_COORDINATES = config["start_coordinates"]
            MAP_CHECK_POINT_LIST = config["check_point_list"]

            sim_car = Car(actions_dict=actions_dict, car_file='./sim_world/envs/Lego-Robot.png', energy=CAR_ENERGY_START, energy_max=CAR_ENERGY_MAX)
            sim_pygame = Simulation(map_file_path=MAP, car=sim_car, start_coordinates=MAP_START_COORDINATES, checkpoints_list=MAP_CHECK_POINT_LIST)
            env = gym.make("Robot_Simulation_Pygame-v2", pygame=sim_pygame)

            # Use MAP as the key for tracking rewards and goals for each map
            if MAP not in map_rewards_goal_rates:
                map_rewards_goal_rates[MAP] = {'rewards': [], 'goals': 0}

        __state = env.reset()
        __state = noise_sensors(__state, noiseConf)
        __state = pp.preprocess(observations=__state, states_list=states_list)
        __state = __convert_3d_to_1d(state_3d=__state, observation_space_num=observation_space_num)
        __action = agent.selectAction(__state)
        __done = False
        __experiences = [{}]
        __reward_sums.append(0.0)

        while (not __done):
            __timesteps += 1
            __experiences[-1]['state'] = __state
            __experiences[-1]['action'] = __action
            __experiences[-1]['done'] = __done

            __new_state, __reward, __done, __info = env.step(__action)
            __new_state = noise_sensors(__new_state, noiseConf)
            __new_state = pp.preprocess(observations=__new_state, states_list=states_list)
            __new_state = __convert_3d_to_1d(state_3d=__new_state, observation_space_num=observation_space_num)

            __new_action = agent.selectAction(__new_state)

            __xp = {}
            __xp['state'] = __new_state
            __xp['reward'] = __reward
            __xp['done'] = __done
            __xp['action'] = __new_action
            __experiences.append(__xp)

            agent.update(__experiences[-2:])

            if (agent.getName() == "SARSA"):
                __action = __new_action
            else:
                __action = agent.selectAction(__new_state)

            __state = __new_state

            __reward_sums[-1] += __reward

            if (__e % 20 == 0):
                env.render()

        __episodesvstimesteps.append([__e, __timesteps])

        total_steps += __timesteps
        total_rewards += __reward_sums[-1]
        if randomMap:
            map_rewards_goal_rates[MAP]['rewards'].append(__reward_sums[-1])
        if __info.get('goal_reached', False):
            goals_reached += 1
            if randomMap:
                map_rewards_goal_rates[MAP]['goals'] += 1
            print(f"Episode {__e + 1}: REACHED GOAL")
            print(f"Episode {__e + 1}: Reward = {__reward_sums[-1]}")

        # Store table data
        if (__e % 50 == 0):
            if (agent.getName() == 'Double Q-Learning'):
                __avg_action_table = np.mean(np.array([agent.actionValueTable_1.copy(), agent.actionValueTable_2.copy()]), axis=0)
                __actionValueTable_history.append(__avg_action_table.copy())
            else:
                __actionValueTable_history.append(agent.actionValueTable.copy())

        if (__e % 100 == 0):
            __title = agent.getName() + ' Episode:' + str(__e)
            logger.debug('%s | reward_sums = \'%s\'', str(__title), str(__reward_sums[-1]))

    avg_steps = total_steps / agent_nEpisodes
    avg_reward = total_rewards / agent_nEpisodes
    goal_rate = (goals_reached / agent_nEpisodes) * 100

    stats = {
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'goal_rate': goal_rate
    }

    # Compute the average reward and goal rate for each map
    for map_key in map_rewards_goal_rates:
        rewards = map_rewards_goal_rates[map_key]['rewards']
        goals = map_rewards_goal_rates[map_key]['goals']
        map_rewards_goal_rates[map_key]['avg_reward'] = sum(rewards) / len(rewards) if rewards else 0
        map_rewards_goal_rates[map_key]['goal_rate'] = (goals / len(rewards)) * 100 if rewards else 0

    return __reward_sums, np.array(__episodesvstimesteps), __actionValueTable_history, stats, map_rewards_goal_rates

def _test_q_table(q_table, env, states_list, agent_nEpisodes):
    """test the given q-table under a greedy policy (argmax)

    Args:
        q_table (np.array): q_table to test
        env (gym.Env): gym Environment to test the agent in
        agent_nEpisodes (int): number of episodes

    Returns:
        dict: map_goal_rates
        float: total_goal_rate
    """
    __observation_space_nums = __get_observation_space_num(env=env, states_list=states_list)

    map_goal_rates = {}
    total_goals = 0
    total_episodes = 0

    for map_name, config in maps_config.items():
        map_goal_rates[map_name] = {'goals': 0, 'episodes': 0}

        MAP = config["path"]
        MAP_START_COORDINATES = config["start_coordinates"]
        MAP_CHECK_POINT_LIST = config["check_point_list"]

        sim_car = Car(actions_dict=actions_dict, car_file='./sim_world/envs/Lego-Robot.png', energy=CAR_ENERGY_START, energy_max=CAR_ENERGY_MAX)
        sim_pygame = Simulation(map_file_path=MAP, car=sim_car, start_coordinates=MAP_START_COORDINATES, checkpoints_list=MAP_CHECK_POINT_LIST)
        env = gym.make("Robot_Simulation_Pygame-v2", pygame=sim_pygame)

        for __e in range(100):  # Test each map for 100 episodes
            __timesteps = 0
            __state = env.reset()

            # preprocessing of the measured values
            __state = pp.preprocess(observations=__state, states_list=states_list)

            __q_table_index = __convert_3d_to_1d(state_3d=__state, observation_space_num=__observation_space_nums)
            __action = np.argmax(q_table[__q_table_index])
            __done = False
            while not __done:
                __timesteps += 1

                __new_state, __reward, __done, __info = env.step(__action)

                # preprocessing of the measured values
                __new_state = pp.preprocess(observations=__new_state, states_list=states_list)

                # transform to 1d-coordinates
                __new_state = __convert_3d_to_1d(state_3d=__new_state, observation_space_num=__observation_space_nums)

                __action = np.argmax(q_table[__new_state])

                __state = __new_state

                if __e % 10 == 0:
                    env.render()

            map_goal_rates[map_name]['episodes'] += 1
            total_episodes += 1
            if __info.get('goal_reached', False):
                map_goal_rates[map_name]['goals'] += 1
                total_goals += 1
                print(f"Episode {__e + 1} on Map {map_name}: REACHED GOAL")

    # Compute the goal rate for each map
    for map_key in map_goal_rates:
        map_goal_rates[map_key]['goal_rate'] = (map_goal_rates[map_key]['goals'] / map_goal_rates[map_key]['episodes']) * 100

    # Compute the total goal rate for all maps
    total_goal_rate = (total_goals / total_episodes) * 100

    return map_goal_rates, total_goal_rate

def __get_observation_space_num(env, states_list):
    """get the number of steps per dimenion of the observation space

    Args:
        env (gym.Env): env to obtain the observation from

    Returns:
        list: number of steps per observation dimension
    """
    # how many states per dimension exist?
    __observation_space_nums = [len(states_list[0]), len(states_list[1]), len(states_list[2])]
    return __observation_space_nums

def __convert_q_table_to_policy(q_table, observation_space_num):
    """convert the q-table to a greedy policy lookup dictionary

    Args:
        q_table (np.array): q-table to create the policy
        observation_space_num (): #TODO

    Returns:
        dict: greedy policy dict {state:action}
    """
    __policy_dict = {}
    for __i, __state in enumerate(q_table):
        __skey = ___convert_1d_to_3d(state_1d=__i, observation_space_num=observation_space_num)
        __skey = str(__skey[0]) + ' ' + str(__skey[1]) + ' ' + str(__skey[2])
        __action = int(np.argmax(__state))
        __policy_dict.update({__skey : __action})
    return __policy_dict

def __store_dict_as_json(dict_data, file_name):
    """save a given dict as json file

    Args:
        dict_data (dict): dict to save
        file_name (str): path of the .json file
    """
    with open(file_name, 'w') as __f:
        json.dump(dict_data, __f)

def __convert_3d_to_1d(state_3d, observation_space_num):
    """convert 3d States to 1d States

    Args:
        state_3d (list): 3d state
        observation_space_num (list): list of stages per observation dimension

    Returns:
        state_1d (int): 1d state
    """
    __x_len = observation_space_num[2]
    __y_len = observation_space_num[1]
    __z_len = observation_space_num[0]

    __x = state_3d[2]
    __y = state_3d[1]
    __z = state_3d[0]
    # ravel
    __state_1d = __x + (__y * __x_len) + (__z * __x_len * __y_len)
    return __state_1d

def ___convert_1d_to_3d(state_1d, observation_space_num):
    """convert 1d States to 3d States

    Args:
        state_1d (int): 1d state
        observation_space_num (list): list of stages per observation dimension

    Returns:
        state_3d (tupel): 3d state
    """
    __x_len = observation_space_num[2]
    __y_len = observation_space_num[1]
    __z_len = observation_space_num[0]

    # unravel
    __x = state_1d % __x_len
    __y = (state_1d // __x_len) % __y_len
    __z = ((state_1d // __x_len) // __y_len) % __z_len

    return (__z, __y, __x)

def read_numpy_data(numpy_file):
    """load numpy data

    Args:
        file (str): path to file

    Returns:
        np.array: loaded data
    """
    __data = np.load(numpy_file + '.npy')
    return __data

def optimize_params(trial, env, nStates, states_list, q_table=None):
        
        """train the agent in the given env

        Args:
            trial: suggests hyper parameter
            env (gym.Env): gym.Env to train in 
            agent (agent): reinforcement agent to train
            file_prefix (strings): #TODO
            file_suffix (strings): #TODO
            q_table (np.array, optional): q_table used for init. Defaults to None.
        """
        alpha = trial.suggest_categorical("alpha", search_space['alpha'])
        gamma = trial.suggest_categorical("gamma", search_space['gamma'])
        logger.info("New set of hyper parameter: alpha=%.2f, gamma=%.2f, epsilon=%.2f", alpha, gamma, policy_epsilon)
        if agentIdx == 0:
            agent = TDL.SARSA(nStates, nActions, alpha,
                            gamma, epsilon=policy_epsilon)
        elif agentIdx == 1:
            agent = TDL.QLearning(nStates, nActions, alpha,
                                gamma, epsilon=policy_epsilon)
        elif agentIdx == 2:
            agent = TDL.DoubleQLearning(
                nStates, nActions, alpha, gamma, epsilon=policy_epsilon
            )
        __observation_space_nums = __get_observation_space_num(env=env, states_list=states_list)
        if (not (q_table is None)):
            agent.actionValueTable = q_table
            logger.info('USE GIVEN Q-TABLE')

        __reward_sums, __evst, __actionValueTable_history, stats = _runExperiment_NStep(agent_nEpisodes=agent_nEpisodes, env=env, agent=agent, states_list=states_list, observation_space_num=__observation_space_nums)
        logger.info('Results of hyper parameter: alpha=%.2f, gamma=%.2f, epsilon=%.2f', alpha, gamma, policy_epsilon)
        logger.info('Training Statistics: Avg Steps = %.2f, Avg Reward = %.2f, Goal Rate = %.2f%%',
                    stats['avg_steps'], stats['avg_reward'], stats['goal_rate'])
        return stats['avg_reward'], stats['goal_rate']/100

def noise_sensors(state, noiseConf):
    state[0]=max(0,state[0]+random.randint(noiseConf['west'][0], noiseConf['west'][1]))
    state[1]=max(0,state[1]+random.randint(noiseConf['north'][0], noiseConf['north'][1]))
    state[2]=max(0,state[2]+random.randint(noiseConf['ost'][0], noiseConf['ost'][1]))
    return state

def plot_rewards(rewards, map_rewards_goal_rates):
    # Berechne den gleitenden Durchschnitt mit einem Fenster von 10 Episoden
    window_size = 10

    # moving average
    rewards_smooth = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    
    # Erstelle das Liniendiagramm des gleitenden Durchschnitts
#    plt.figure(figsize=(12, 6))
#    plt.plot(rewards_smooth, label='Moving Average of Rewards', color='orange')
#    plt.xlabel('Episode')
#    plt.ylabel('Sum of Rewards')
#    plt.title('Moving Average of Rewards in Each Episode')
#    plt.legend()
#    plt.grid(True)
#    plt.show()

    return rewards_smooth


def plot_goal_rates(map_rewards_goal_rates, overall_goal_rate):
    map_names = list(map_rewards_goal_rates.keys())
    goal_rates = [data['goal_rate'] for data in map_rewards_goal_rates.values()]
    
    # Add overall goal rate to the data
    map_names.append('Overall')
    goal_rates.append(overall_goal_rate)
    
    plt.figure(figsize=(12, 6))
    plt.bar(map_names, goal_rates, color='blue')
    plt.xlabel('Map')
    plt.ylabel('Goal Rate (%)')
    plt.title('Goal Rate per Map and Overall')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def setup_random_map():
    map_names = list(maps_config.keys())
    selected_map = random.choice(map_names)
    config = get_map_config(selected_map)
    if not config:
        logger.error("Map config not found for map: %s", selected_map)
        sys.exit(1)
    return config

#def check_actions(action):
#    if(action == 0):
    
################################################################
###                          M A I N                         ###
################################################################

if __name__ == "__main__":

    
    
    parser = argparse.ArgumentParser(description='Verarbeite agentIdx und states_listIdx.')

    # Füge Argumente hinzu
    parser.add_argument('--agentIdx', type=int, default=0)
    parser.add_argument('--statesIdx', type=int, default=1)
    parser.add_argument('--filePfx', default="")

    # Argumente parsen
    args = parser.parse_args()

    ### get commandline parameters: 1st arg: agent; 2nd arg state_list ###
    agentIdx = args.agentIdx
    states_listIdx = args.statesIdx
    file_prefix = args.filePfx

    ROOT_FILE_PATH = "../model_storage/zE04/"
    current_datetimestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(ROOT_FILE_PATH + current_datetimestamp):
        os.makedirs(ROOT_FILE_PATH + current_datetimestamp)
    CURRENT_FILE_PATH = ROOT_FILE_PATH + current_datetimestamp + "/"

    
    # logger config
    # https://docs.python.org/3/library/logging.html#logrecord-attributes   
    logger = logger.createLogger(config="../logging.conf.json", logger_name="__main__", logfile=CURRENT_FILE_PATH + 'logfile.log')

    logger.info('START SIMULATION')
    pp = DH()
    ###############################################################
    ############################ SETUP ############################

    TRAIN_MODEL = True
    RETRAIN_MODEL = False
    TEST_MODEL = True
    RUN_MODEL = True
    OPTIMIZE = False

    #noiseConf = {"north": [0, 0], "west": [0, 0], "ost": [0, 0]}
    noiseConf = {"north": [-6, -5], "west": [-1, 1], "ost": [-1, 1]}

    # manual path
    # CURRENT_FILE_PATH = ""

    ######################### ENVIRONMENT #########################

    config = get_map_config("race3")
    if config:
        MAP = config["path"]
        MAP_START_COORDINATES = config["start_coordinates"]
        MAP_CHECK_POINT_LIST = config["check_point_list"]
    else:
        logger.error("Map config not found")
        sys.exit(1)

    CAR_ENERGY_START = 2000
    CAR_ENERGY_MAX = 2000

    # States & Actions
    # states_list = [['west'], ['north'], ['east']]
    # states_list = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    # 353, 575, 474
    states_list = [
        [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2]],
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4]],
        [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3]],
    ]
    compute_states = lambda x,y,z: len(x)*len(y)*len(z)
    nStates = compute_states(*states_list[states_listIdx])

    actions_dict = {
        0: {'speed' : 15, 'energy' : -2},
        1: {'angle' : -45, 'energy' : -15},
        2: {'angle' : 45, 'energy' : -15},
        3: {'angle' : -15, 'energy' : -5},
        4: {'angle' : 15, 'energy' : -5},
        5: {'speed' : 5, 'energy' : -1}
    }
    nActions = len(actions_dict)

    sim_car = Car(actions_dict = actions_dict, car_file = './sim_world/envs/Lego-Robot.png', energy = CAR_ENERGY_START, energy_max = CAR_ENERGY_MAX)
    sim_pygame = Simulation(map_file_path = MAP, car = sim_car,  start_coordinates = MAP_START_COORDINATES, checkpoints_list = MAP_CHECK_POINT_LIST)
    env = gym.make("Robot_Simulation_Pygame-v2", pygame = sim_pygame)

    ###############################################################
    ######################### MODEL ###############################

    agent_exerciseID = 0
    agent_nExperiments = 1
    agent_nEpisodes = 100

    # Agent
    hyperparameter = [
        {'alpha': 0.1, 'gamma': 0.9 },    # SARSA
        {'alpha': 0.006, 'gamma': 0.75 },   # Q
        {'alpha': 0.006, 'gamma': 0.8 },    # QQ
    ]
    agent_alpha = hyperparameter[agentIdx]['alpha'] #qq 0.006 # 2st 0.005 # 1st 0.1
    agent_gamma = hyperparameter[agentIdx]['gamma'] #qq 0.8 # 2st 0.78 # 1st 0.9
    agent_n_steps = 5 # 5
    
    # Policy
    policy_epsilon = 0.1

    epsilon_decay = 0.9 + (0.00004 / (agent_nEpisodes / 100))
    if epsilon_decay > 1:
        epsilon_decay = 1

    # env.render()

    # Agent
    agents = [
        TDL.SARSA(nStates, nActions, agent_alpha, agent_gamma, epsilon=policy_epsilon),
        TDL.QLearning(nStates, nActions, agent_alpha, agent_gamma, epsilon=policy_epsilon),
        TDL.DoubleQLearning(nStates, nActions, agent_alpha, agent_gamma, epsilon=policy_epsilon)
    ]

    agent = agents[agentIdx]
    logger.info("AGENT '%s' SELECTED", str(agent.getName()))
    file_suffix = "_" + agent.getName() + "_" + current_datetimestamp

    ######################### HYPERPARAMETER OPTIMIZATION #########################
    if OPTIMIZE:
        logger.info("OPTIMIZE PARAMETER [%s]", str(CURRENT_FILE_PATH))
        objective = partial(
            optimize_params,
            env=env,
            nStates=nStates,
            states_list=states_list[states_listIdx],
        )
        # 1st: search_space = {"alpha": [0.01, 0.05, 0.1, 0.15, 0.2], "gamma": [0.8, 0.85, 0.9, 0.95, 0.99]}
        # 2st: search_space = {"alpha": [0.001, 0.005, 0.01], "gamma": [0.75, 0.78, 0.8, 0.82, 0.85]} ## Q-Learning
        search_space = {"alpha": [0.006, 0.007, 0.008, 0.009], "gamma": [0.75, 0.78, 0.8, 0.82, 0.85]}
        #search_space = {"alpha": [0.01, 0.05], "gamma": [0.99, 0.8]}
        study = optuna.create_study(directions=["maximize","maximize"],
            sampler=optuna.samplers.GridSampler(search_space))
        study.optimize(objective, show_progress_bar=True)
        try:
            opt_params = np.save(CURRENT_FILE_PATH + file_prefix + "optimized-params" + file_suffix + ".npy",
                [study.best_trials],
            )
            fig = vis.plot_contour(study, target_name="Goal rate", target=lambda t: t.values[1])
            #fig.update_layout({"title": ""+ str(agents[agentIdx].getName()) +" - Race 3 - "+agent_nEpisodes+" episodes - " + "575"})
            fig.show()
            fig.write_image(CURRENT_FILE_PATH + file_prefix + "optimized_params" + file_suffix + ".jpg", format="jpg")
            fig.write_json(CURRENT_FILE_PATH + file_prefix + "optimized_params" + file_suffix + ".json")
            logger.info("FINISHED OPTIMIZING")

        except RuntimeError as e:
                logger.info(e)
                logger.info("Could not find goal with parameter from search_space.")
        # Set optimized paramter for traininig
        
    ######################### AGENT TRAIN #########################
    if (TRAIN_MODEL):
        logger.info("TRAIN AGENT [%s]", str(CURRENT_FILE_PATH))
        train_model(env=env, agent=agent, file_path=CURRENT_FILE_PATH, file_prefix=file_prefix, file_suffix=file_suffix, states_list=states_list[states_listIdx])
        logger.info("FINISH TRAINING OF AGENT")

    ######################### AGENT TRAIN #########################

    if (RETRAIN_MODEL):
        q_data_file = CURRENT_FILE_PATH + file_prefix + 'q-table' + file_suffix
        rewards_file = CURRENT_FILE_PATH + file_prefix + 'reward_sums' + file_suffix
        if hasattr(agent.policy, "epsilon"):
          agent.policy.epsilon = 0
        # manual path
        #q_data_file = ""
        #rewards_file= ""

        # load rewards and q-table
        rewards = read_numpy_data(rewards_file)
        logger.debug('rewards loaded = \'%s\'', str(rewards))
        logger.info('LOADING TRAININGSDATA \'%s\'', str(q_data_file))
        q_data = read_numpy_data(numpy_file=q_data_file)

        logger.info('RETRAIN AGENT WITH \'%s\' [%s]', str(q_data_file), str(CURRENT_FILE_PATH))
        train_model(env=env, agent=agent, states_list=states_list[states_listIdx], file_path=CURRENT_FILE_PATH, file_prefix=file_prefix, file_suffix=file_suffix, q_table=q_data)
        logger.info('FINISH RETRAINING OF AGENT')

    ######################### AGENT TEST ##########################

    if (TEST_MODEL):
        logger.info("TEST AGENT [%s]", str(CURRENT_FILE_PATH))
        test_model(env=env, agent=agent, states_list=states_list[states_listIdx], file_path=CURRENT_FILE_PATH, file_prefix=file_prefix, file_suffix=file_suffix)
        logger.info("FINISH TESTING OF AGENT")

    ######################### AGENT RUN ###########################

    if (RUN_MODEL):
        policy_file = CURRENT_FILE_PATH + file_prefix + 'policy' + file_suffix + '.json'
        # manual path
        #policy_file = "../model_storage/policy_SARSA.json"

        logger.info('LOADING POLICY \'%s\'', str(policy_file))
        policy = load_policy(policy_as_json=policy_file)
        logger.info('RUN AGENT WITH \'%s\' [%s]', str(policy_file), str(CURRENT_FILE_PATH))
        run_model(env=env, policy=policy, states_list=states_list[states_listIdx])
        logger.info('FINISH RUNNING OF AGENT')
    
    ###############################################################

    logger.info('EXIT SIMULATION')
