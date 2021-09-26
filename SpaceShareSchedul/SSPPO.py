from Base.Parameters import Parameters

from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import random
import collections
import torch.optim as optim
from Schedul import Reward
from torch.distributions import Categorical

class DRLParameters(object):
    memory_size = 10**4
    batch_size = 6
    epochs = int(memory_size/batch_size)*10
    train_step = 3 #几个回合 进行一次学习
    buffer_mini_size = 1000
    gamma = 0.8 #0.99 #累计回报
    learning_rate = 0.0001 # 学习率
    updata_target_step = train_step * 3

    mark = 0

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(Actor, self).__init__()
        self.device = device
        '''
        网络层
        '''
        self.input_layer = nn.Linear(input_dim, 60 * input_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(2, 8):
            self.hidden_layers.append(nn.Linear(int(60 * input_dim / (i - 1)), int(60 * input_dim / i)))
        self.output_layer = nn.Linear(int(60 * input_dim / i), output_dim)

    def forward(self, X):
        lengths = []
        x = []
        try:
            for i in range(len(X)):
                lengths.append(len(X[i]))
                for j in range(len(X[i])):
                    x.append(X[i][j])
        except TypeError as e:
            print(e)
        x = torch.tensor(x, dtype=torch.float32)
        y = self.forward_item(x)
        out_y = []
        c_index = 0
        for l in lengths:
            out_y.append(F.softmax(y[c_index:c_index + l].reshape(-1), dim=0))
            c_index += l
        return out_y


    def forward_item(self, x):
        '''
        前向传播
        :param x:
        :return:
        '''
        x = F.leaky_relu(self.input_layer(x))
        for h_l in self.hidden_layers:
            x = F.leaky_relu(h_l(x))
        x = F.relu(self.output_layer(x))
        return x

class Critic(nn.Module):

    def __init__(self, input_dim, output_dim, device='cpu'):
        nn.Module.__init__(self)
        self.device = device
        '''
        网络层
        '''
        self.input_layer = nn.Linear(input_dim, 60 * input_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(2, 8):
            self.hidden_layers.append(nn.Linear(int(60 * input_dim / (i - 1)), int(60 * input_dim / i)))
        self.output_layer = nn.Linear(int(60 * input_dim / i), output_dim)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        x = F.leaky_relu(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = F.leaky_relu(hidden_layer(x))
        x = self.output_layer(x)
        return x
    pass


class SimplePPOMemory(object):
    def __init__(self):
        '''
        存储相应数据 以循环队列的方式
        '''
        self.buffer = collections.deque(maxlen=DRLParameters.memory_size)
        self.position = 0
        self.size = 0

    def push(self, data):
        self.buffer.append(data)

    def pushs(self, datas):
        self.buffer.extend(datas)

    def sample(self, num):
        mini_batch = random.sample(self.buffer, num)
        s_, a_, a_value_, r_, sp_, rs_, done_ = [], [], [], [], [], [], []
        s1_ = []
        for item in mini_batch:
            s, a, av, r, s_prime, rs, done = item
            s_.append(s[0])
            s1_.append(s[1])
            a_.append(a)
            a_value_.append(av)
            r_.append(r)
            sp_.append(s_prime)
            rs_.append(rs)
            done_.append(done)
        return s_, s1_, a_, a_value_, r_, sp_, rs_, done_

    def handle_track(self, tracks):
        l = len(tracks)
        if l < 2:
            return False

        s = tracks[l-1]['s']
        a = tracks[l-1]['a']
        r = tracks[l-1]['r']
        s_prime = tracks[l-1]['s']
        a_value = tracks[l-1]['a_value']
        rs = r
        done = 1
        self.push((s, a, a_value, r,s_prime, rs, done))
        i = l - 2
        while i >= 0:
            s = tracks[i]['s']
            a = tracks[i]['a']
            a_value = tracks[i]['a_value']
            r = tracks[i]['r']
            s_prime = tracks[i+1]['s']
            rs = r + DRLParameters.gamma * rs
            done = 0
            self.push((s, a, a_value, r, s_prime, rs, done))
            i -= 1
        return True

    def buffSize(self):
        # 返回池子的大小
        return len(self.buffer)

    def clear(self):
        return self.buffer.clear()

import time

class Schedule(object):

    def __init__(self):
        self.actor = Actor(5,1)
        self.critic = Critic(4,1)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DRLParameters.learning_rate, eps=1e-8)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DRLParameters.learning_rate, eps=1e-8)

        self._sample_limit = (0.1, 1)
        self.sample_decay = (self._sample_limit[1] - self._sample_limit[0]) / int(
            DRLParameters.epochs / DRLParameters.train_step)
        self.sample_rand_rate = self._sample_limit[1]

        self.training = False
        self.localtime = time.localtime()

        self.save_flod = 'SimplePPO'
        self.clip_range = 0.2

    def train(self, memory):
        # if self.sample_rand_rate > self._sample_limit[0]:
        #     self.sample_rand_rate -= (self.sample_decay*5)
        #     print(self.sample_rand_rate)
        gamma = DRLParameters.gamma
        batch_size = DRLParameters.batch_size

        eps = self.clip_range
        for _ in range(round(1 * memory.buffSize() / batch_size)):
            # index = np.random.choice(np.arange(memory.bufferSize), batch_size, replace=False)
            s, s1, a, a_value, r, sp, rs, done_ = memory.sample(DRLParameters.batch_size)
            v_target = torch.tensor(rs, dtype=torch.float32)
            s1_t = torch.tensor(s1, dtype=torch.float32)
            v = self.critic(s1_t)
            delta = v_target - v
            advantage = delta.detach()
            a_t = torch.tensor(1, dtype=torch.float32)
            s0 = s
            c_a_value = self.actor(s0)
            c_a_value_a = []
            for index, ca in zip(a, c_a_value):
                c_a_value_a.append(ca[index].reshape(-1))
            c_a_value_t = torch.cat(c_a_value_a, dim=0)
            a_value_p = torch.tensor(a_value, dtype=torch.float32)
            ratio = c_a_value_t / a_value_p
            surr1 = ratio * advantage
            clipped_ratio = torch.clamp(ratio, 1. - eps, 1. + eps)
            surr2 = clipped_ratio * advantage

            pol_surr = -torch.min(surr1, surr2).mean()

            loss = nn.MSELoss()
            value_loss = loss(v, v_target)
            # error2 = (v - v_target).pow(2).mean()

            self.actor_optimizer.zero_grad()
            pol_surr.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

    def selectAction(self, x):
        randv = random.random()
        training = self.training
        if training:
            # a = self.dqn(x)
            # return Universal.sample(a[0].detach().numpy())
            a = self.actor(x)
            dist = Categorical(probs=a[0])
            a_out = dist.sample()
            a_value = a[0][a_out]
            return a_out, a_value
            # if randv > self.sample_rand_rate:
            #     # 按模型来
            #
            #     a_out = a[0].argmax().item()
            #     a_value = a[0][a_out]
            #     return a_out, a_value
            # else:
            #     # 随机
            #     a_out = random.randint(0, len(x[0]) - 1)
            #     a_value = a[0][a_out]
            #     return a_out, a_value
        else:
            a = self.actor(x)
            a_out = a[0].argmax().item()
            a_value = a[0][a_out]
            return a_out, a_value

    def task2mc(self, cluster, index=0):
        '''
        决策 tasks mcs
        遍历task
        单次决策： 决策 + 状态记录
        决策：x
        :param cluster:
        :return:
        '''
        if len(cluster.task_buffer) == 0:
            return None
        task = cluster.task_buffer[index]
        X, x2,  choose_mc_list = self.getS(cluster, index)
        if len(X) == 0:
            return None
        a_out, a_value = self.selectAction([X])
        try:
            return {choose_mc_list[a_out]:[{
                'task': task,
                's':(X, x2),
                'a':a_out,
                'av':a_value,
            },]}
        except IndexError as e:
            print(e)

        pass

    def getS(self, cluster, index=0):
        task = cluster.task_buffer[index]
        total = {}

        total['busy_time'] = []
        for mc in cluster.machine_list:
            total['busy_time'].append(mc.busy_time)
        total['busy_time_std'] = np.array(total['busy_time']).std()
        total['busy_time_mean'] = np.array(total['busy_time']).mean()
        mean_wait_time = cluster.mean_wait_task_time
        X = []
        choose_mc_list = []
        for mc in cluster.machine_list:
            if not mc.matchTask(task):
                continue
            x = []
            x.append(len(cluster.task_buffer))
            x.append(total['busy_time_std'])
            x.append(total['busy_time_mean'])
            x.append(mc.busy_time)
            x.append(mc.tryExcuteTask(task))

            X.append(x)
            choose_mc_list.append(mc)
            for i in x:
                if np.isnan(i):
                    print('???')
                    raise Exception('NaN')

        x2 = []
        x2.append(len(cluster.task_buffer))
        x2.append(total['busy_time_std'])
        x2.append(total['busy_time_mean'])
        x2.append(mean_wait_time)

        return X,x2, choose_mc_list

    def saveModel(self, path=None):
        if path == None:
            t_s = time.strftime('%Y-%m-%d-%H-%M', self.localtime)
            file_name = '{}-{}.pkl'.format(t_s, self.actor.__class__.__name__)
            path = '{}/{}/{}'.format(Parameters.SavePath, self.save_flod, file_name)
            torch.save(self.actor.state_dict(), path)

            t_s = time.strftime('%Y-%m-%d-%H-%M', self.localtime)
            file_name = '{}-{}.pkl'.format(t_s, self.critic.__class__.__name__)
            path = '{}/{}/{}'.format(Parameters.SavePath, self.save_flod, file_name)
            torch.save(self.critic.state_dict(), path)
        pass

    def loadModel(self, t_s, path=None):
        if path == None:
            file_name = '{}-{}.pkl'.format(t_s, self.actor.__class__.__name__)
            path = '{}/{}/{}'.format(Parameters.SavePath, self.save_flod, file_name)
            self.actor.load_state_dict(torch.load(path))

            file_name = '{}-{}.pkl'.format(t_s, self.critic.__class__.__name__)
            path = '{}/{}/{}'.format(Parameters.SavePath, self.save_flod, file_name)
            self.critic.load_state_dict(torch.load(path))

        pass

