import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


from gym.wrappers import GrayScaleObservation #Stack Frames (our AI will be able
#to see. GrayScaleObservation to convert images in gray = less data to process)
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

#print(SIMPLE_MOVEMENT) #7 different types of actions

#Create environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

#print(env.action_space)
print(env.observation_space.shape) #Observation we get

#plt.imshow(env.reset())
#plt.show()
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
env = GrayScaleObservation(env, keep_dim=True) #Keep dim to be able to do stack the frames
print("after grayscaling",env.reset().shape)
#plt.imshow(env.reset())
#plt.show()

#Wrap inside the dummy environment
env = DummyVecEnv([lambda: env]) #we return env in a list
print("new shape : ", env.reset().shape)

#Stack the frames
env = VecFrameStack(env, 4, channels_order='last') #We choose to stack 4 frames.
print(env.reset().shape) #(1,x,y,4) because we stacked 4 frames

state, reward, done, info = env.step([env.action_space.sample()]) #in a list because we vectorized our environment
plt.figure(figsize=(6,4))
for i in range(state.shape[3]):
    plt.subplot(1,4,i+1)
    plt.imshow(state[0][:,:,i])
plt.show()
