from Base.Parameters import Parameters, DRLParameters

from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import random
import collections
import torch.optim as optim
from Schedul import Reward
from Tool.Universal import Universal

class DDQNOne(nn.Module):
    '''
    s?:   可用 cpu、bw标准差   任务cpu、bw 占a——mc 百分比  | 所有工作负载的均值+标准差 | mc数
    a?:   mc 可用cpu，bw 百分比   mc 工作量： 工作负载：Σ任务的资源A*剩余时间 /对应mc资源的总值  |均值 标准差
    mc 中的任务的 资源需求均值 标准差  剩余时间均值
    out？： Qv
    '''
    def __init__(self, input_dim, output_dim, device = 'cpu'):
        super(DDQNOne, self).__init__()
        self.device = device
        '''
        网络层
        '''
        fact = 30
        self.input_layer = nn.Linear(input_dim, fact*input_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(2,4):
            self.hidden_layers.append(nn.Linear(int(fact*input_dim/(i-1)), int(fact*input_dim/i)))
        self.output_layer = nn.Linear(int(fact*input_dim/i), output_dim)



    #X = [ [[]],[] ]
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
            out_y.append(y[c_index:c_index + l].reshape(-1))
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
        x = F.leaky_relu(self.output_layer(x))
        return x



    #[[[],[],[]]]


class DDQNOneMemory(object):
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
        s_, a_, r_, sp_, done_ = [], [], [], [], []
        for item in mini_batch:
            s, a, r, s_prime, done = item
            s_.append(s)
            a_.append(a)
            r_.append(r)
            sp_.append(s_prime)
            done_.append(done)
        return s_, a_, r_, sp_, done_

    def handle_track(self, tracks):
        l = len(tracks)
        if l < 2:
            return False

        s = tracks[l-1]['s']
        a = tracks[l-1]['a']
        r = tracks[l-1]['r']
        s_prime = tracks[l-1]['s']
        done = 1
        self.push((s,a,r,s_prime, done))
        i = l - 2

        # total_r = r
        # gae_r = r
        # total_gae_r = gae_r

        while i >= 0:
            s = tracks[i]['s']
            a = tracks[i]['a']
            r = tracks[i]['r']
            s_prime = tracks[i+1]['s']
            done = 0
            self.push((s, a, r, s_prime, done))
            i -= 1

            # total_r += r
            # gae_r = gae_r*DRLParameters.gamma + r
            # total_gae_r += gae_r
        return True
        # return total_r, total_gae_r, l

    def buffSize(self):
        # 返回池子的大小
        return len(self.buffer)

    def clear(self):
        return self.buffer.clear()




import time

class Schedule(object):
    ModelSaveName = 'SDDQN'
    def __init__(self):
        self.dqn = DDQNOne(15,1)
        self.target = DDQNOne(15,1)
        # self.gamma = DRLParameters.gamma
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=DRLParameters.learning_rate, eps=1e-8)
        self.updataTarget()
        '''
        动作选择
        '''
        self._sample_limit = (0, 1)
        self.sample_decay = (self._sample_limit[1]-self._sample_limit[0])/int(5000/DRLParameters.train_step)
        self.sample_rand_rate = self._sample_limit[1]

        self.training = True
        self.localtime = time.localtime()

    def train(self, memory):
        if self.sample_rand_rate > self._sample_limit[0]:
            self.sample_rand_rate -= (self.sample_decay*5)
            print(self.sample_rand_rate)
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

    def getReward(self, **kwargs):
        cluster = kwargs['cluster']
        task = kwargs['task']
        mc = kwargs['mc']
        wait = cluster.env.clock - task.load_time
        dut = mc.tryExcuteTask(task)
        ext = max(wait - task.duration + dut, 0)

        return Reward.reward4Cluster_Makespan(wait, dut, ext, task.duration)
        pass
    def updataTarget(self):
        self.target.load_state_dict(self.dqn.state_dict())

    def selectAction(self, x):
        randv = random.random()
        training = self.training
        if training:
            # a = self.dqn(x)
            # return Universal.sample(a[0].detach().numpy())
            if randv > self.sample_rand_rate:
                # 按模型来
                a = self.dqn(x)
                return a[0].argmax().item()
            else:
                # 随机
                a = random.randint(0, len(x[0])-1)
                return a
        else:
            a = self.dqn(x)
            return a[0].argmax().item()

    def getS(self, cluster, index=0):
        task = cluster.task_buffer[index]
        total = {}

        total['busy_time'] = []
        total['busy_price'] = []
        total['a_time'] = []
        total['a_price'] = []
        total['sla'] = []
        for mc in cluster.machine_list:
            total['busy_time'].append(mc.busy_time)
            total['busy_price'].append(mc.busy_expend)
        total['busy_time_std'] = np.array(total['busy_time']).std()
        total['busy_time_mean'] = np.array(total['busy_time']).mean()
        total['busy_price_std'] = np.array(total['busy_price']).std()
        total['busy_price_mean'] = np.array(total['busy_price']).mean()
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


            x.append(total['busy_price_std'])
            x.append(total['busy_price_mean'])
            x.append(mc.price)
            # x.append(mc.taskExpend(task))
            x.append(max(mc.busy_time - task.load_time + cluster.env.clock, 0))

            total['sla'].append(max(mc.busy_time - task.load_time + cluster.env.clock, 0))
            total['a_time'].append(mc.tryExcuteTask(task))
            total['a_price'].append(mc.price)
            X.append(x)
            choose_mc_list.append(mc)
        for x in X:
            x.append(np.array(total['a_time']).mean())
            x.append(np.array(total['a_time']).std())
            x.append(np.array(total['a_price']).mean())
            x.append(np.array(total['a_price']).std())
            x.append(np.array(total['sla']).mean())
            x.append(np.array(total['sla']).std())
            for i in x:
                if np.isnan(i):
                    print('???')
                    raise Exception('NaN')
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
            path = '{}/{}/{}/{}'.format(Parameters.SavePath, self.ModelSaveName, t_s, file_name)
            torch.save(self.dqn.state_dict(), path)
        pass

    def loadModel(self, filename, path=None):
        if path == None:
            path = '{}/{}'.format(Parameters.SavePath, filename)
            self.dqn.load_state_dict(torch.load(path))
            self.updataTarget()
        pass

    def getFilenamePrefix(self):
        t_s = time.strftime('%Y-%m-%d-%H-%M', self.localtime)
        return {
            'file_name': '{}-{}.pkl'.format(t_s, self.dqn.__class__.__name__),
            'file_folder': self.dqn.__class__.__name__,
        }
        pass
    def getTimeStr(self):
        return time.strftime('%Y-%m-%d-%H-%M', self.localtime)
