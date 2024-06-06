import json
from ev3_robot_car import EV3Car as Car

def load_policy(policy_as_json):
    """load the policy json file

    Args:
        path_json (str): path to json file

    Returns:
        dict: policy dict: {state:action}
    
    """
    __policy = {}
    with open(policy_as_json, 'r') as __f:
      __policy = json.load(__f)
    return __policy

def run_model(car, policy):
    """run the car with the given policy

    Args:
        car (EV3Car): car to drive given the policy
        policy (dict): policy
    """
    __done = False
    while (__done == False):        
        # observe the current state
        __observations = car.observe()

        # preprocessing of the measured values
        __state = preprocessing_observations9(observations=__observations)

        # select the best action; based on state
        __action = greedy(policy=policy, state=__state)
        car.action(__action)
            
def greedy(policy, state):
    '''
    Greedy policy
    return the best action corresponding to the state
    '''
    __key = str(state[0]) + ' ' + str(state[1]) + ' ' + str(state[2])
    return policy[__key]

def preprocessing_observations(observations):
    __car_size = 24 #cm
    __obs_discrete = []

    for __observation in observations:
        if(__observation < (__car_size * 1.8)):
            __value = 1 # obstacle detected
            if(__observation < (__car_size * 0.6)):
                __value = 0 # close obstacle detected
        else:
            __value = 2 
        __obs_discrete.append(__value)

    return __obs_discrete

def preprocessing_observations5(observations):
    __car_size = 24 #cm
    __obs_discrete = []
    __resize_factor = 1
    for __observation in observations:
        if(__observation < (5 + __car_size * __resize_factor)):
            __value = 0 # obstacle detected
        elif(__observation < (10 + __car_size * __resize_factor)):
            __value = 1 # obstacle detected
        elif(__observation < (15 + __car_size * __resize_factor)):
            __value = 2 # obstacle detected
        elif(__observation < (20 + __car_size * __resize_factor)):
            __value = 3 # obstacle detected
        elif(__observation < (25 + __car_size * __resize_factor)):
            __value = 4 
        __obs_discrete.append(__value)

    return __obs_discrete

def preprocessing_observations7(observations):
    __car_size = 24 #cm
    __obs_discrete = []
    __resize_factor = 1
    for __observation in observations:
        if(__observation < (5 + __car_size * __resize_factor)):
            __value = 1 # obstacle detected
        elif(__observation < (10 + __car_size * __resize_factor)):
                __value = 2 # obstacle detected
        elif(__observation < (15 + __car_size * __resize_factor)):
                __value = 3 # obstacle detected
        elif(__observation < (20 + __car_size * __resize_factor)):
                __value = 4 # obstacle detected
        elif(__observation < (25 + __car_size * __resize_factor)):
                __value = 5 # obstacle detected
        elif(__observation < (30 + __car_size * __resize_factor)):
                __value = 6 # obstacle detected
        else:
            __value = 0 
        __obs_discrete.append(__value)

    return __obs_discrete

def preprocessing_observations9(observations):
    __car_size = 24 #cm
    __obs_discrete = []
    __resize_factor = 1
    for __observation in observations:
        if(__observation < (5 + __car_size * __resize_factor)):
            __value = 1 # obstacle detected
        elif(__observation < (10 + __car_size * __resize_factor)):
                __value = 2 # obstacle detected
        elif(__observation < (15 + __car_size * __resize_factor)):
                __value = 3 # obstacle detected
        elif(__observation < (20 + __car_size * __resize_factor)):
                __value = 4 # obstacle detected
        elif(__observation < (25 + __car_size * __resize_factor)):
                __value = 5 # obstacle detected
        elif(__observation < (30 + __car_size * __resize_factor)):
                __value = 6 # obstacle detected
        elif(__observation < (35 + __car_size * __resize_factor)):
                __value = 7 # obstacle detected
        elif(__observation < (40 + __car_size * __resize_factor)):
                __value = 8 # obstacle detected
        else:
            __value = 0 
        __obs_discrete.append(__value)

    return __obs_discrete

################################################################
###                          M A I N                         ###
################################################################

if __name__ == '__main__':
    RUN_MODEL = True

    ######################### ENVIRONMENT #########################

    # trained policy
    #policy_file = '../../model_storage/policy_SARSA.json'
    # local policy
    policy_file = 'model_storage/policy_SARSA.json'
    
    actions_dict = {
        0: {'speed' : 20},
        1: {'angle' : -45},
        2: {'angle' : 45},
        3: {'speed' : -20}
    }
    
    ev3 = Car(actions=actions_dict)

    if (RUN_MODEL):
        policy = load_policy(policy_as_json=policy_file)
        run_model(car=ev3, policy=policy)