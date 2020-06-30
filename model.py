# -*- coding:utf-8 -*-
"""
@author: liuying
@license: Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
@contact: liuying49@baidu.com
@file: model.py
@time: 2020/6/30 10:14 上午
"""
import os
import numpy as np
import gym

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放


class PolicyModel(parl.Model):
    def __init__(self, act_dim, init_logvar=-1):
        ######################################################################
        ######################################################################
        #
        # 2. 请配置model结构
        #
        ######################################################################
        ######################################################################
        self.act_dim = act_dim
        self.fc1 = layers.fc(size=100, act='relu')
        self.fc2 = layers.fc(size=act_dim, act='tanh')

        self.logvars = layers.create_parameter(
            shape=[act_dim],
            dtype='float32',
            default_initializer=fluid.initializer.ConstantInitializer(
                init_logvar))

    def policy(self, obs):
        ######################################################################
        ######################################################################
        #
        # 3. 请组装policy网络
        #
        ######################################################################
        ######################################################################
        hid1 = self.fc1(obs)
        means = self.fc2(hid1)
        logvars = self.logvars()
        return means, logvars

    def sample(self, obs):
        means, logvars = self.policy(obs)
        print(means, logvars)
        sampled_act = means + (
                layers.exp(logvars / 2.0) *  # stddev
                layers.gaussian_random(shape=(self.act_dim,), dtype='float32'))
        return sampled_act

class ValueModel(parl.Model):
    def __init__(self):
        ######################################################################
        ######################################################################
        #
        # 4. 请配置model结构
        #
        ######################################################################
        ######################################################################
        self.fc1 = layers.fc(size=100, act='relu')
        self.fc2 = layers.fc(size=1, act=None)

class ValueModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(ValueModel, self).__init__()
        hid1_size = obs_dim * 10
        hid3_size = 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.lr = 1e-2 / np.sqrt(hid2_size)

        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=1)

    def value(self, obs):
        hid1 = self.fc1(obs)
        V = self.fc2(hid1)
        V = layers.squeeze(V, axes=[])
        return V


class BipedalWalkerModel(parl.Model):
    def __init__(self, obs_dim, act_dim, init_logvar=-1.0):
        self.policy_model = PolicyModel(act_dim, init_logvar)
        self.value_model = ValueModel(obs_dim, act_dim)

    #         self.policy_lr = self.policy_model.lr
    #         self.value_lr = self.value_model.lr

    def policy(self, obs):
        return self.policy_model.policy(obs)

    def policy_sample(self, obs):
        return self.policy_model.sample(obs)

    def value(self, obs):
        return self.value_model.value(obs)