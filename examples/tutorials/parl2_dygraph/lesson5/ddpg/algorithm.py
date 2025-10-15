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
import torch.nn.functional as F
from copy import deepcopy


class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """ DDPG algorithm
        
        Args:
            model (parl.Model): actor and critic 的前向网络.
                                model 必须实现 get_actor_params() 方法.
            gamma (float): reward的衰减因子.
            tau (float): self.target_model 跟 self.model 同步参数 的 软更新参数
            actor_lr (float): actor 的学习率
            critic_lr (float): critic 的学习率
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.get_critic_params(), lr=critic_lr)

    def predict(self, obs):
        """ 使用 self.model 的 actor model 来预测动作
        """
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 用DDPG算法更新 actor 和 critic
        """
        critic_loss = self._critic_learn(obs, action, reward, next_obs, terminal)
        actor_loss = self._actor_learn(obs)
        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            target_Q = self.target_model.value(next_obs, self.target_model.policy(next_obs))
            terminal = terminal.float()
            target_Q = reward + ((1. - terminal) * self.gamma * target_Q)

        current_Q = self.model.value(obs, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()

    def _actor_learn(self, obs):
        actor_loss = -self.model.value(obs, self.model.policy(obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def sync_target(self, decay=None):
        """ 软更新 target_model 参数 """
        if decay is None:
            decay = 1.0 - self.tau
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(decay * target_param.data + (1.0 - decay) * param.data)
