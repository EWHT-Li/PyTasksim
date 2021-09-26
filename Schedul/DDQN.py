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
        self.input_layer = nn.Linear(input_dim, 60*input_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(2,8):
            self.hidden_layers.append(nn.Linear(int(60*input_dim/(i-1)), int(60*input_dim/i)))
        self.output_layer = nn.Linear(int(60*input_dim/i), output_dim)



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
        x = F.relu(self.output_layer(x))
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

    # def handleTrackWithrecode(self, track, recode):
    #     pass
    '''
    def extractRewardForRecode(self, start, dur, recodes):
        recodes_l = len(recodes)
        i = 0
        while i < recodes_l:
            item=recodes[i]
            if start < item['time']:
                i -= 1
                break
            i += 1
        reward = 0
        c_start = start
        while i < recodes_l-1:
            item = recodes[i]
            item_n = recodes[i+1]
            if item['time'] >= start + dur:
                break
            if start + dur >= item_n['time']:
                reward += (item['balance']/item['mc_num'])*(item_n['time']-c_start)
                c_start = item_n['time']
            else:
                reward += (item['balance'] / item['mc_num']) * (start + dur - item['time'])
                c_start = start + dur
            i += 1
        return reward
'''



import time

class Schedule(object):
    def __init__(self):
        self.dqn = DDQNOne(16,1)
        self.target = DDQNOne(16,1)
        # self.gamma = DRLParameters.gamma
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=DRLParameters.learning_rate, eps=1e-8)
        self.updataTarget()
        '''
        动作选择
        '''
        self._sample_limit = (0.1, 1)
        self.sample_decay = (self._sample_limit[1]-self._sample_limit[0])/int(DRLParameters.epochs/DRLParameters.train_step)
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
            for s in Parameters.BaseSource: #为了反映集群的状态 4
                x.append(total_source_p[s + '_std'])  # 集群资源可用比标准差   一台机器CPU用了10% 那可用比就是90%
                x.append(total_source_p[s + '_mean'])  # 均值
            for s in Parameters.BaseSource: # 用任务资源对应mc的百分比，是为了泛化性，百分比更容易复现 2
                x.append(task.need_source[s] / mc.source[s])  # task 在 对应mc 上的资源占比
            x.append(task.duration)  # 非ex_source model 任务的持续时间
            for s in Parameters.BaseSource: # 2
                x.append(a_mc_source[mc][s + '_p']) # mc可用资源百分比
            x.append(cluster.env.clock - task.load_time)  # 超时时间 原本是还考虑了sla的目标，所以加入了任务超时的东西，不过目前暂时只考虑负载均衡 1
            mc_task_source = {}
            for s in Parameters.BaseSource:
                mc_task_source[s + '_p'] = []
            mc_task_source['remain_time'] = []
            for m_t in mc.task_list: # 收集对应mc中 所执行任务的信息
                for s in Parameters.BaseSource:
                    mc_task_source[s + '_p'].append(m_t.need_source[s] / mc.source[s]) #每个在执行任务对应所占资源百分比
                mc_task_source['remain_time'].append(m_t.duration - cluster.env.clock + m_t.start_time) # 任务剩余时间
            if len(mc.task_list) != 0:
                for s in Parameters.BaseSource: # 6
                    x.append(np.array(mc_task_source[s + '_p']).std()) # 以标准差和均值的方式 对资源特征和剩余时间特征进行一定程度上的抽象反映
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
        #************************
        # total_a_source ={}
        # for s in Parameters.BaseSource:
        #     total_a_source[s] = 0
        #
        # task_list = cluster.task_buffer[:]
        # if len(task_list) == 0:
        #     return None
        # a_mc_source ={}
        # # 对集群资源进行统计 生成a_mc_source & total_a_source
        # for mc in cluster.machine_list:
        #     a_mc_source[mc] = {}
        #     for s in Parameters.BaseSource:
        #         total_a_source[s] += mc.a_source[s]
        #         a_mc_source[mc][s] = mc.a_source[s]
        #     a_mc_source[mc]['task'] = []
        #
        # # 找出这批任务主导资源与任务
        # min_task = None
        # min_task_pd = None
        # min_property = None
        # i = 0
        # min_index = 0
        # for task in task_list:
        #     task_pd = 0
        #     task_property = None
        #     for s in Parameters.BaseSource:
        #         t_task_pd = task.need_source[s] / total_a_source[s]
        #         if t_task_pd > task_pd:
        #             task_pd = t_task_pd
        #             task_property = s
        #     if min_task_pd == None or min_task_pd > task_pd:
        #         min_task_pd = task_pd
        #         min_task = task
        #         min_property = task_property
        #         min_index = i
        #     i += 1
        #
        # index = min_index
        #*************************

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
