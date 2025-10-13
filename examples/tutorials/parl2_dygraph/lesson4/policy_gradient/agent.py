#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#-*- coding: utf-8 -*-

import parl
import torch
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm):
        super(Agent, self).__init__(algorithm)

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        prob = self.alg.predict(obs_tensor)
        prob_np = prob.detach().cpu().numpy()
        act = np.random.choice(len(prob_np), 1, p=prob_np)[0]  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        prob = self.alg.predict(obs_tensor)
        act = int(torch.argmax(prob).item())  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        """ 根据训练数据更新一次模型参数
        """

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        act_tensor = torch.tensor(act, dtype=torch.int64)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        loss = self.alg.learn(obs_tensor, act_tensor, reward_tensor)
        return float(loss)
