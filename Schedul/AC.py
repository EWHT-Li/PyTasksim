from Base.Parameters import Parameters, DRLParameters

from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import random
import collections
import torch.optim as optim
from Schedul import Reward
from torch.distributions import Categorical
import time
from torch.autograd import Variable


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
                out_y.append(y[c_index:c_index + l].reshape(-1))
                c_index += l
        except Exception as e:
            print(e)
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

class SimpleA2CMemory(object):
    def __init__(self):
        '''
        存储相应数据 以循环队列的方式
        '''
        self.buffer = []
        self.position = 0
        self.size = 0

    def push(self, data):
        self.buffer.append(data)

    def pushs(self, datas):
        self.buffer.extend(datas)

    def sample(self, num=0):
        mini_batch = self.buffer
        return mini_batch
        # s_, a_, a_value_, r_, sp_, rs_, done_ = [], [], [], [], [], [], []
        # s1_ = []
        # for item in mini_batch.reverse():
        #     s, a, av, r, s_prime, rs, done = item
        #     s_.append(s[0])
        #     s1_.append(s[1])
        #     a_.append(a)
        #     a_value_.append(av)
        #     r_.append(r)
        #     sp_.append(s_prime)
        #     rs_.append(rs)
        #     done_.append(done)
        # return s_, s1_, a_, a_value_, r_, sp_, rs_, done_

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
        # self.push((s, a, a_value, r,s_prime, rs, done))
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


class Schedule(object):

    def __init__(self):
        self.actor = Actor(16,1)
        self.critic = Critic(6,1)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DRLParameters.learning_rate, eps=1e-8)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DRLParameters.learning_rate, eps=1e-8)

        self._sample_limit = (0.1, 1)
        self.sample_decay = (self._sample_limit[1] - self._sample_limit[0]) / int(
            DRLParameters.epochs / DRLParameters.train_step)
        self.sample_rand_rate = self._sample_limit[1]

        self.training = False
        self.localtime = time.localtime()

        self.save_flod = 'A2C'
        self.clip_range = 0.2

    def train(self, memory):
        gamma = DRLParameters.gamma
        batch_size = DRLParameters.batch_size
        tau = 1
        entropy_coef = 0.01

        eps = self.clip_range
        mini_batch = memory.sample(DRLParameters.batch_size)
        R = None
        value_loss = 0
        value_prime = None
        gae = 0
        policy_loss = 0
        for item in reversed(mini_batch):
            s, a, av, r, s_prime, rs, done = item
            if R == None:
                value_prime = self.critic(s_prime[1])
                R = value_prime.data
            value = self.critic(s[1])
            R = gamma * R + r
            advantage = R - value
            value_loss = value_loss + 0.5 * advantage.pow(2)
            delta_t = r + gamma * value_prime.data - value.data
            gae = gae * gamma * tau + delta_t


            c_a_value = self.actor([s[0]])



            prob = F.softmax(c_a_value[0], dim=0)
            log_prob = F.log_softmax(c_a_value[0], dim=0)
            entropy = - (log_prob * prob).sum()

            policy_loss = policy_loss - log_prob[a] * Variable(gae) - entropy_coef * entropy
            pass

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def selectAction(self, x):
        randv = random.random()
        training = self.training
        if training:
            # a = self.dqn(x)
            # return Universal.sample(a[0].detach().numpy())
            a = self.actor(x)
            a = F.softmax(a[0], dim=0)
            dist = Categorical(probs=a)
            a_out = dist.sample()
            a_value = a[a_out]
            return a_out, a_value
        else:
            a = self.actor(x)
            a = F.softmax(a[0], dim=0)
            a_out = a.argmax().item()
            a_value = a[a_out]
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
        total_source_p ={} # 每台机器对应的可用资源百分比
        for s in Parameters.BaseSource:
            total_source_p[s+'_p'] = []
        a_mc_source = {} # 统计mc的资源（可用，_p可用百分比， _total总资源）
        # 对集群资源进行统计 生成a_mc_source & total_a_source
        for mc in cluster.machine_list:
            a_mc_source[mc] = {}
            for s in Parameters.BaseSource:
                a_mc_source[mc][s] = mc.a_source[s]
                total_source_p[s+'_p'].append(mc.a_source[s]/mc.source[s])
                a_mc_source[mc][s+'_p'] = mc.a_source[s]/mc.source[s]
                a_mc_source[mc][s + '_total'] = mc.source[s]
            a_mc_source[mc]['task'] = []
        for s in Parameters.BaseSource:
            total_source_p[s+'_std'] = np.array(total_source_p[s+'_p']).std() # 集群内各项资源对应的标准差
            total_source_p[s + '_mean'] = np.array(total_source_p[s + '_p']).mean()

        task = cluster.task_buffer[index]
        X = []

        choose_mc_list = []
        for mc in cluster.machine_list:
            if not mc.matchTask(task):
                continue
            x = []
            for s in Parameters.BaseSource:
                x.append(total_source_p[s + '_std'])  # 集群资源使用比标准差
                x.append(total_source_p[s + '_mean'])  # 均值
            for s in Parameters.BaseSource:
                x.append(task.need_source[s] / mc.source[s])  # task 在 对应mc 上的资源占比
            x.append(task.duration)  # 非ex_source model
            for s in Parameters.BaseSource:
                x.append(a_mc_source[mc][s + '_p'])
            x.append(cluster.env.clock - task.load_time)  # 超时时间
            mc_task_source = {}
            for s in Parameters.BaseSource:
                mc_task_source[s + '_p'] = []
            mc_task_source['remain_time'] = []
            for m_t in mc.task_list:
                for s in Parameters.BaseSource:
                    mc_task_source[s + '_p'].append(m_t.need_source[s] / mc.source[s])
                mc_task_source['remain_time'].append(m_t.duration - cluster.env.clock + m_t.start_time)
            if len(mc.task_list) != 0:
                for s in Parameters.BaseSource:
                    x.append(np.array(mc_task_source[s + '_p']).std())
                    x.append(np.array(mc_task_source[s + '_p']).mean())
                x.append(np.array(mc_task_source['remain_time']).std())
                x.append(np.array(mc_task_source['remain_time']).mean())
            else:
                for s in Parameters.BaseSource:
                    x.append(0)
                    x.append(0)
                x.append(0)
                x.append(0)
            X.append(x)
            choose_mc_list.append(mc)

        x2 = []
        for s in Parameters.BaseSource:
            x2.append(total_source_p[s + '_std'])  # 集群资源使用比标准差
            x2.append(total_source_p[s + '_mean']/Parameters.BaseSourceLimit[s][1])  # 均值
        for s in Parameters.BaseSource:
            x2.append(task.need_source[s] / Parameters.BaseSourceLimit[s][1])


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