import gymnasium as gym
from gridworld import CliffWalkingWapper

env = gym.make('CliffWalking-v0')
env = CliffWalkingWapper(env)
env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    state, reward, done, _,info = env.step(action)
    if done:
        break