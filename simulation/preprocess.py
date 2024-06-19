import logging, logging.config
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

if __name__ == "__main__":
    opt_sarsa = np.load("model_storage/2024-06-01_10-33-05/world3optimized-params_SARSA_2024-06-01_10-33-05.npy", allow_pickle=True)
    opt_q = np.load("model_storage/2024-06-01_10-33-13/world3optimized-params_Q-Learning_2024-06-01_10-33-13.npy", allow_pickle=True)
    opt_dq = np.load("model_storage/2024-06-01_10-33-24/world3optimized-params_Double Q-Learning_2024-06-01_10-33-24.npy", allow_pickle=True)

    opt_q