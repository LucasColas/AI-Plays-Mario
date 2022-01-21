import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


from gym.wrappers import FrameStack, GrayScaleObservation #Stack Frames (our AI will be able
#to see. GrayScaleObservation to convert images in gray = less data to process)
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

#print(SIMPLE_MOVEMENT) #7 different types of actions

#Create environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

#print(env.action_space)
print(env.observation_space.shape) #Observation we get

#Running the game
"""
done = True
for step in range(100000): #Loop through each frame in the game
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()
"""


"""
Preprocessing
"""

#Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT) #Wrapping environment to use simple actions

#Grayscaling
env = GrayScaleObservation(env, keep_dim=True)
print("after grayscaling",env.reset().shape)
