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

import copy
import parl
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            gamma (float): reward的衰减因子
            lr (float): learning_rate，学习率.
        """
        # checks
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)
        self.gamma = gamma
        self.lr = lr
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, obs):
        with torch.no_grad():
            return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        obs_tensor = torch.FloatTensor(obs)
        action_tensor = torch.LongTensor(action).squeeze(-1)
        reward_tensor = torch.FloatTensor(reward)
        next_obs_tensor = torch.FloatTensor(next_obs)
        terminal_tensor = torch.FloatTensor(terminal)

        pred_values = self.model(obs_tensor)
        action_dim = pred_values.shape[-1]
        action_onehot = F.one_hot(action_tensor, num_classes=action_dim)
        pred_value = torch.sum(pred_values * action_onehot, dim=1, keepdim=True)

        with torch.no_grad():
            max_v = self.target_model(next_obs_tensor).max(1, keepdim=True)[0]
            target = reward_tensor + (1 - terminal_tensor) * self.gamma * max_v
        loss = self.mse_loss(pred_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
