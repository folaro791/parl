#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import gymnasium as gym
from gridworld import CliffWalkingWapper
from agent import QLearningAgent
import time

def run_episode(env, agent,render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    obs=obs[0]

    while True:
        action = agent.sample(obs)  # 根据算法选择一个动作
        if render:
            render_q_table(agent.Q, total_steps,(obs, action))  # 终端打印Q表并高亮将要更新的Q值
            time.sleep(0.4)  # 暂停方便观察
        next_obs, reward, done, _ , _= env.step(action)  # 与环境进行一个交互
        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1  # 计算step数
        if done:
            break
    return total_reward, total_steps

def render_q_table(q_table, steps,action_step=None):
        print("Q-table step:",steps)
        highlight = action_step
        RED='\033[91m'
        RESET = '\033[0m'

        for y in range(4): 
            up_row = []
            second_row = []
            third_row = []
            down_row = []
            for x in range(12):
                s = y * 12 + x
                q = q_table[s]
                # 判断是否高亮
                def fmt(val, a, label):
                    if highlight == (s, a):
                        return f" {RED}{label}{val:>7.2f}{RESET} "
                    else:
                        return f" {label}{val:>7.2f} "
                up_row.append(fmt(q[0], 0, 'U:'))
                second_row.append(fmt(q[1], 1, 'R:'))
                third_row.append(fmt(q[2], 2, 'D:'))
                down_row.append(fmt(q[3], 3, 'L:'))
            print(''.join(up_row))
            print(''.join(second_row))
            print(''.join(third_row))
            print(''.join(down_row))
            print()
        print("-"*120)

def test_episode(env, agent):
    total_reward = 0
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)
    obs = env.reset()[0]
    step = 1
    while True:
        env.render()
        action = agent.predict(obs)  # greedy
        render_q_table(agent.Q, step,(obs, action))
        time.sleep(1)
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        step += 1
        
        if done:
            print('test reward = %.1f' % (total_reward))
            break


def main():
    # env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
    # env = FrozenLakeWapper(env)

    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWapper(env)

    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)


    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent,render=False)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))

    # 训练结束，查看算法效果
    test_episode(env, agent)


if __name__ == "__main__":
    main()
