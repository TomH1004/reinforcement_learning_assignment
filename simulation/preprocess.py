import logging, logging.config
import logger
import numpy as np

class DiscretizeHelper:
    
    def __init__(self, logger):
        self.logger=logger

    # Parallel to this method, there is a method in reinforcement_learning_assignment\robot\ev3\main_robot_micpy.py
    def preprocessing_observations(self, observations, states_list):
        # states_list = [['west'], ['north'], ['east']]
        __car_X = 50
        __car_size = 24  # cm
        __resize_factor = __car_X / __car_size
        __obs_discrete = []

        for __observation in observations:
            if __observation < ((__car_size * __resize_factor) * 1.8):
                # obstacle detected
                __value = states_list[1][1]
                if __observation < ((__car_size * __resize_factor) * 0.6):
                    # close obstacle detected
                    __value = states_list[1][0]
            else:
                # no obstacle detected
                __value = states_list[1][2]
            __obs_discrete.append(__value)
        self.logger.debug("OBSERVATION DISCRETE '%s'", str(__obs_discrete))
        return __obs_discrete


    def preprocessing_observations_5(self, observations, states_list):
        # states_list = [['west'], ['north'], ['east']]
        __car_X = 50
        __car_size = 24  # cm
        __resize_factor = __car_X / __car_size
        __obs_discrete = []

        for __observation in observations:
            if __observation < (5 + (__car_size * __resize_factor)): 
                # obstacle detected
                __value = states_list[1][0]
            elif (__observation < (10 * __resize_factor + __car_size)):
                # close obstacle detected
                __value = states_list[1][1]
            elif __observation < (15 * __resize_factor + __car_size):
                # obstacle detected
                __value = states_list[1][2]
            elif __observation < (20 * __resize_factor + (__car_size)) and len(states_list[0])>3 and len(states_list[2])>3:
                # obstacle detected
                __value = states_list[1][3]
            elif (len(states_list[0])>3 and len(states_list[2])>3):
                # no obstacle detected
                __value = states_list[1][4]
            else:
                __value = states_list[1][2]
            __obs_discrete.append(__value)
        self.logger.debug("OBSERVATION DISCRETE '%s'", str(__obs_discrete))
        return __obs_discrete


    def preprocessing_observations_7(self, observations, states_list):
        # states_list = [['west'], ['north'], ['east']]
        __car_X = 50
        __car_size = 24  # cm
        __resize_factor = __car_X / __car_size
        __obs_discrete = []

        for __observation in observations:
            if __observation < (5 + (__car_size * __resize_factor)):
                # obstacle detected
                __value = states_list[1][0]
            elif __observation < (10 * __resize_factor + __car_size):
                # close obstacle detected
                __value = states_list[1][1]
            elif __observation < (15 * __resize_factor + __car_size):
                # obstacle detected
                __value = states_list[1][2]
            elif __observation < (20 * __resize_factor + (__car_size)):
                # obstacle detected
                __value = states_list[1][3]
            elif __observation < (25 * __resize_factor + (__car_size)):
                # obstacle detected
                __value = states_list[1][4]
            elif __observation < (30 * __resize_factor + (__car_size)):
                # obstacle detected
                __value = states_list[1][5]
            else:
                # no obstacle detected
                __value = states_list[1][6]
            __obs_discrete.append(__value)
        self.logger.debug("OBSERVATION DISCRETE '%s'", str(__obs_discrete))
        return __obs_discrete
    
    def preprocessing_observations_9(self, observations, states_list):
        # states_list = [['west'], ['north'], ['east']]
        __car_X = 50
        __car_size = 24  # cm
        __resize_factor = __car_X / __car_size
        __obs_discrete = []

        for __observation in observations:
            if __observation < (5 + (__car_size * __resize_factor)):
                # obstacle detected
                __value = states_list[1][0]
            elif __observation < (10 * __resize_factor + __car_size):
                # close obstacle detected
                __value = states_list[1][1]
            elif __observation < (15 * __resize_factor + __car_size):
                # obstacle detected
                __value = states_list[1][2]
            elif __observation < (20 * __resize_factor + (__car_size)):
                # obstacle detected
                __value = states_list[1][3]
            elif __observation < (25 * __resize_factor + (__car_size)):
                # obstacle detected
                __value = states_list[1][4]
            elif __observation < (30 * __resize_factor + (__car_size)):
                # obstacle detected
                __value = states_list[1][5]
            elif __observation < (35 * __resize_factor + (__car_size)):
                # obstacle detected
                __value = states_list[1][6]
            elif __observation < (40 * __resize_factor + (__car_size)):
                # obstacle detected
                __value = states_list[1][7]
            else:
                # no obstacle detected
                __value = states_list[1][8]
            __obs_discrete.append(__value)
        self.logger.debug("OBSERVATION DISCRETE '%s'", str(__obs_discrete))
        return __obs_discrete


    def preprocess(self, observations, states_list):
        
        if len(states_list[0])==3:
            return self.preprocessing_observations(observations, states_list)
        if len(states_list[0])==5:
            return self.preprocessing_observations_5(observations, states_list)
        if len(states_list[0])==7:
            return self.preprocessing_observations_7(observations, states_list)
        if len(states_list[0])==9:
            return self.preprocessing_observations_9(observations, states_list)

if __name__ == "__main__":
    opt_sarsa = np.load("model_storage/2024-06-01_10-33-05/world3optimized-params_SARSA_2024-06-01_10-33-05.npy", allow_pickle=True)
    opt_q = np.load("model_storage/2024-06-01_10-33-13/world3optimized-params_Q-Learning_2024-06-01_10-33-13.npy", allow_pickle=True)
    opt_dq = np.load("model_storage/2024-06-01_10-33-24/world3optimized-params_Double Q-Learning_2024-06-01_10-33-24.npy", allow_pickle=True)

    opt_q