import gymnasium as gym
from gridworld import CliffWalkingWapper

env = gym.make('CliffWalking-v0')
env = CliffWalkingWapper(env)
env.reset()
generation = 1
print("当前状态:",36)
print("第",generation,"代出发了")
acumulated_reward = 0
reward_list =[]
while True:

    #env.render()
    action = env.action_space.sample()
    state, reward, done, _,info = env.step(action)
    print("采取动作:",action," 进入状态:",state,"  获得奖励:",reward)
    acumulated_reward += reward
    if reward == -100:
        reward_list.append(acumulated_reward+99)
        print("总共得到奖励:",acumulated_reward)
        acumulated_reward = 0
        generation+=1
        print("第",generation,"代出发了")
        print("当前状态:",36)

    if done:
        reward_list.append(acumulated_reward)
        print("总共得到奖励:",acumulated_reward)
        print("终于到达终点了!")
        print("每一代得到的奖励:",reward_list)
        print("最小:",min(reward_list))
        print("总共获得reward:",sum(reward_list))
        print("一共经历了",generation,"代")
        print("平均得到奖励:",sum(reward_list)/generation)
        break