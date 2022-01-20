import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#print(SIMPLE_MOVEMENT) #7 different types of actions

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT) #Wrapping environment to use simple actions
#print(env.action_space)
env.observation_space.shape
