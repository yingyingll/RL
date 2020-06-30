# -*- coding:utf-8 -*-
"""
@author: liuying
@license: Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
@contact: liuying49@baidu.com
@file: train.py.py
@time: 2020/6/30 10:28 上午
"""

import numpy as np
import scipy.signal
import argparse
import gym
import numpy as np
import parl
from parl.utils import logger, action_mapping
from parl.utils.rl_utils import calc_gae, calc_discount_sum_rewards
from parl.algorithms import PPO
from agent import BiPedalwalkerAgent
from model import BipedalWalkerModel


ACTOR_LR = 0.0002   # Actor网络更新的 learning rate
CRITIC_LR = 0.001   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward

__all__ = ['Scaler']


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0
        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.cnt = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)
        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.cnt = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = (
                (self.means * self.cnt) + (new_data_mean * n)) / (self.cnt + n)
            self.vars = (((self.cnt * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) /
                         (self.cnt + n) - np.square(new_means))
            self.vars = np.maximum(
                0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.cnt += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means




def run_train_episode(env, agent, scaler):
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    env.reset()
    while True:
        obs = obs.reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        obs = obs.astype('float32')
        observes.append(obs)

        action = agent.policy_sample(obs)
        action = np.clip(action, -1.0, 1.0)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        action = action.reshape((1, -1)).astype('float32')
        env.render()
        actions.append(action)

        obs, reward, done, _ = env.step(np.squeeze(action))
        rewards.append(reward)
        step += 1e-3  # increment time step feature

        if done:
            break

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype='float32'), np.concatenate(unscaled_obs))


def run_evaluate_episode(env, agent, scaler):
    obs = env.reset()
    rewards = []
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while True:
        obs = obs.reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        obs = (obs - offset) * scale  # center and scale observations
        obs = obs.astype('float32')

        action = agent.policy_predict(obs)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        obs, reward, done, _ = env.step(np.squeeze(action))
        env.render()
        rewards.append(reward)

        step += 1e-3  # increment time step feature

        if done:
            break
    return np.sum(rewards)


def collect_trajectories(env, agent, scaler, episodes):
    trajectories, all_unscaled_obs = [], []
    for e in range(episodes):
        obs, actions, rewards, unscaled_obs = run_train_episode(
            env, agent, scaler)
        trajectories.append({
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
        })
        all_unscaled_obs.append(unscaled_obs)
    # update running statistics for scaling observations
    scaler.update(np.concatenate(all_unscaled_obs))
    return trajectories


def build_train_data(trajectories, agent):
    train_obs, train_actions, train_advantages, train_discount_sum_rewards = [], [], [], []
    for trajectory in trajectories:
        pred_values = agent.value_predict(trajectory['obs'])

        # scale rewards
        scale_rewards = trajectory['rewards'] * (1 - 0.995)

        discount_sum_rewards = calc_discount_sum_rewards(
            scale_rewards, 0.995).astype('float32')

        advantages = calc_gae(scale_rewards, pred_values, 0, 0.995,
                              0.98)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-6)
        advantages = advantages.astype('float32')

        train_obs.append(trajectory['obs'])
        train_actions.append(trajectory['actions'])
        train_advantages.append(advantages)
        train_discount_sum_rewards.append(discount_sum_rewards)

    train_obs = np.concatenate(train_obs)
    train_actions = np.concatenate(train_actions)
    train_advantages = np.concatenate(train_advantages)
    train_discount_sum_rewards = np.concatenate(train_discount_sum_rewards)

    return train_obs, train_actions, train_advantages, train_discount_sum_rewards

if __name__ == '__main__':
    # 创建飞行器环境
    import gym

    env = gym.make("BipedalWalker-v2")
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_dim += 1  # add 1 to obs dim for time step feature

    scaler = Scaler(obs_dim)

    model = BipedalWalkerModel(obs_dim, act_dim, -1.0)
    alg = PPO(model, act_dim=act_dim, policy_lr=ACTOR_LR, value_lr=CRITIC_LR)
    agent = BiPedalwalkerAgent(alg, obs_dim, act_dim, 0.003, loss_type='CLIP')

    # parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
    collect_trajectories(env, agent, scaler, episodes=5)

    test_flag = 0
    total_steps = 0
    while total_steps < 100000000:
        trajectories = collect_trajectories(
            env, agent, scaler, episodes=5)
        total_steps += sum([t['obs'].shape[0] for t in trajectories])
        total_train_rewards = sum([np.sum(t['rewards']) for t in trajectories])

        train_obs, train_actions, train_advantages, train_discount_sum_rewards = build_train_data(
            trajectories, agent)

        policy_loss, kl = agent.policy_learn(train_obs, train_actions,
                                             train_advantages)
        value_loss = agent.value_learn(train_obs, train_discount_sum_rewards)

        if total_steps // TEST_EVERY_STEPS >= test_flag:  # 每隔一定step数，评估一次模型
            while total_steps // TEST_EVERY_STEPS >= test_flag:
                test_flag += 1

            eval_reward = run_evaluate_episode(env, agent, scaler)
            logger.info('Steps {}, Evaluate reward: {}'.format(
                total_steps, eval_reward))  # 打印评估的reward


