class DiscretizeHelper:
    
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
    
