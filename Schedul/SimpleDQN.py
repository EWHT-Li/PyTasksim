from Base.Parameters import Parameters, DRLParameters

from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import random
import collections
import torch.optim as optim
from Schedul import Reward


class DDQN12(nn.Module):
    '''
    s?:   可用 cpu、bw标准差   任务cpu、bw 占a——mc 百分比  | 所有工作负载的均值+标准差 | mc数
    a?:   mc 可用cpu，bw 百分比   mc 工作量： 工作负载：Σ任务的资源A*剩余时间 /对应mc资源的总值  |均值 标准差
    mc 中的任务的 资源需求均值 标准差  剩余时间均值
    out？： Qv
    '''
    def __init__(self, input_dim, output_dim, device = 'cpu'):
        super(DDQN12, self).__init__()
        self.device = device
        '''
        网络层
        '''
        self.input_layer = nn.Linear(input_dim, 16*input_dim)
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(16 * input_dim, 8 * input_dim))
        self.hidden_layers.append(nn.Linear(8 * input_dim, 4 * input_dim))
        self.hidden_layers.append(nn.Linear(4 * input_dim, 2 * input_dim))
        self.hidden_layers.append(nn.Linear(2 * input_dim, 1 * input_dim))
        self.output_layer = nn.Linear(1 * input_dim, output_dim)



    #X = [ [[]],[] ]
    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.forward_item(X)


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










import time

class Schedule(object):
    def __init__(self):
        self.dqn = DDQN12(14,3)
        self.target = DDQN12(14,3)
        # self.gamma = DRLParameters.gamma
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=DRLParameters.learning_rate, eps=1e-8)
        self.updataTarget()
        '''
        动作选择
        '''
        self._sample_limit = (0.01, 1)
        self.sample_decay = (self._sample_limit[1]-self._sample_limit[0])/int(DRLParameters.epochs/DRLParameters.train_step)
        self.sample_rand_rate = self._sample_limit[1]

        self.training = True
        self.localtime = time.localtime()

    def train(self, memory):
        if self.sample_rand_rate > self._sample_limit[0]:
            self.sample_rand_rate -= self.sample_decay
        gamma = DRLParameters.gamma
        optimizer = self.optimizer
        for i in range(50):
            s, a, r, sp, done = memory.sample(DRLParameters.batch_size)
            q = self.dqn(s)
            tq = self.target(sp)

            q_1 = []
            for index, value in zip(a,q):
                q_1.append(value[index:index+1])
            q_2 = torch.cat(q_1)
            r_2 = torch.tensor(r, dtype=torch.float32)
            done_2 = torch.tensor(done, dtype=torch.float32)
            tq_1 = []
            for i in tq:
                tq_1.append(i.max().item())
            tq_2 = torch.tensor(tq_1, dtype=torch.float32)

            expected_state_action_values = r_2 + gamma * tq_2 * (1-done_2)
            loss = F.mse_loss(q_2, expected_state_action_values)
            # loss = F.smooth_l1_loss(q_2, expected_state_action_values)


            optimizer.zero_grad()
            loss.backward()
            # 采用梯度截断Clip策略来避免梯度爆炸，将梯度约束在某一个区间内
            # for param in self.dqn.parameters():
            #     param.grad.data.clamp_(-1, 1)
            optimizer.step()
        pass

    def updataTarget(self):
        self.target.load_state_dict(self.dqn.state_dict())

    def selectAction(self, x):
        randv = random.random()
        training = self.training
        if training:
            if randv > self.sample_rand_rate:
                # 按模型来
                a = self.dqn(x)
                return a[0].argmax().item()
            else:
                # 随机
                a = random.randint(0, 2)
                return a
        else:
            a = self.dqn(x)
            return a[0].argmax().item()

    def getS(self, cluster, index=0):
        X = []
        choose_mc_list = []
        task = cluster.task_buffer[index]
        for s in Parameters.BaseSource:
            X.append(task.need_source[s]/Parameters.BaseSourceLimit[s][1])
        for mc in cluster.machine_list:
            for s in Parameters.BaseSource:
                X.append(mc.a_source[s]/mc.source[s])
                X.append(mc.a_source[s]/Parameters.BaseSourceLimit[s][1])
            choose_mc_list.append(mc)
        return X, choose_mc_list

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
        X, choose_mc_list = self.getS(cluster, index)
        if len(X) == 0:
            return None
        a_out = self.selectAction([X])
        try:
            return {choose_mc_list[a_out]:[{
                'task':task,
                's':X,
                'a':a_out,
            },]}
        except IndexError as e:
            print(e)

        pass

    def saveModel(self, path=None):
        if path == None:
            t_s = time.strftime('%Y-%m-%d-%H-%M', self.localtime)
            file_name = '{}-{}.pkl'.format(t_s, self.dqn.__class__.__name__)
            path = '{}/{}/{}'.format(Parameters.SavePath, self.dqn.__class__.__name__, file_name)
            torch.save(self.dqn.state_dict(), path)
        pass

    def loadModel(self, filename, path=None):
        if path == None:
            path = '{}/{}'.format(Parameters.SavePath, filename)
            self.dqn.load_state_dict(torch.load(path))
            self.updataTarget()
        pass
