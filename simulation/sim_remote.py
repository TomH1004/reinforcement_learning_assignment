import os
import gym
import pygame
import sys
from sim_world.envs.car_0.ev3_sim_car import SimCar as Car
from sim_world.envs.pygame_0.ev3_sim_pygame_2d_V2 import PyGame2D as Simulation

path_to_main = ".\simulation"

os.chdir(os.getcwd() + path_to_main)

def run_manual_control():
    pygame.init()

    # Environment setup
    MAP = './sim_world/race_tracks/1.PNG'
    MAP_START_COORDINATES = (90, 550)
    MAP_CHECK_POINT_LIST= [(290, 550), (670, 250), (1210, 160)]
    
    CAR_ENERGY_START = 1000
    CAR_ENERGY_MAX = 1000

    actions_dict = {
        0: {'speed': 20, 'energy': -10},
        1: {'angle': -45, 'energy': -10},
        2: {'angle': 45, 'energy': -10},
        3: {'speed': -20, 'energy': -10},
        4: {'speed': 0, 'angle': 0, 'energy': 0}
    }

    sim_car = Car(actions_dict=actions_dict, car_file='./sim_world/envs/Lego-Robot.png', 
                  energy=CAR_ENERGY_START, energy_max=CAR_ENERGY_MAX)
    sim_pygame = Simulation(map_file_path=MAP, car=sim_car, start_coordinates=MAP_START_COORDINATES, 
                            checkpoints_list=MAP_CHECK_POINT_LIST)

    env = gym.make("Robot_Simulation_Pygame-v2", pygame=sim_pygame)

    # Main loop
    running = True
    env.reset()
    env.step(4)
    env.render()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # Increase speed
                elif event.key == pygame.K_RIGHT:
                    action = 1  # Turn left
                elif event.key == pygame.K_LEFT:
                    action = 2  # Turn right
                elif event.key == pygame.K_DOWN:
                    action = 3  # Decrease speed
                else:
                    pygame.quit()

                # Execute action in the environment
                obs, reward, done, info = env.step(action)
                print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
                env.render()

                if done:
                    print("Resetting environment")
                    env.reset()

    pygame.quit()

if __name__ == "__main__":
    run_manual_control()
