import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#print(SIMPLE_MOVEMENT) #7 different types of actions

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT) #Wrapping environment to use simple actions
#print(env.action_space)
print(env.observation_space.shape) #Observation we get

done = True
for step in range(100000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()     
