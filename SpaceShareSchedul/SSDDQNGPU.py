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
    def __init__(self, input_dim, output_dim, device = None): # torch.device('cpu')
        super(DDQNOne, self).__init__()
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(self.device)
        '''
        网络层
        '''
        fact = 30
        self.input_layer = nn.Linear(input_dim, fact*input_dim)#.to(device)
        self.hidden_layers = nn.ModuleList()
        for i in range(2,4):
            self.hidden_layers.append(nn.Linear(int(fact*input_dim/(i-1)), int(fact*input_dim/i)))
        # self.hidden_layers.to(device)
        self.output_layer = nn.Linear(int(fact*input_dim/i), output_dim)#.to(device)
        self.to(device)



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
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = self.forward_item(x).cpu()
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

    def init_weights(self, m):
    	# 对于全连接层
        if type(m) == nn.Linear:
        	# 利用正态分布的方法来初始化参数
            torch.nn.init.kaiming_normal_(m.weight)
            # m.bias.data.fill_将bias直接初始化为0.0（float类型）
            m.bias.data.fill_(0.0)
        # # 对于卷积层
        # if type(m) == nn.Conv2d:
        #     torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')



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
    def __init__(self, device=None): # torch.device('cpu')
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.dqn = DDQNOne(22-4-3-1,1)
        self.target = DDQNOne(22-4-3-1,1)
        # self.gamma = DRLParameters.gamma
        self.dqn.apply(self.dqn.init_weights)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=DRLParameters.learning_rate, eps=1e-8)
        self.updataTarget()
        '''
        动作选择
        '''
        self._sample_limit = (0.1, 1)
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
        all_loss = 0
        recycle = 25
        for i in range(recycle):
            s, a, r, sp, done = memory.sample(DRLParameters.batch_size)
            q = self.dqn(s)
            tq = self.target(sp)

            q_1 = []
            for index, value in zip(a,q):
                q_1.append(value[index:index+1])
            q_2 = torch.cat(q_1)
            r_2 = torch.tensor(r, dtype=torch.float32).to(self.device)
            done_2 = torch.tensor(done, dtype=torch.float32).to(self.device)
            tq_1 = []
            for i in tq:
                tq_1.append(i.max().item())
            tq_2 = torch.tensor(tq_1, dtype=torch.float32).to(self.device)

            expected_state_action_values = r_2 + gamma * tq_2 * (1-done_2)
            q_2 = q_2.to(self.device)
            loss = F.mse_loss(q_2, expected_state_action_values)
            # loss = F.smooth_l1_loss(q_2, expected_state_action_values)


            optimizer.zero_grad()
            loss.backward()
            # 采用梯度截断Clip策略来避免梯度爆炸，将梯度约束在某一个区间内
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            all_loss += loss.cpu().detach().numpy()

        return all_loss/recycle

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
        # self.target.load_state_dict(self.dqn.state_dict())
        d = self.dqn.state_dict()
        t = self.target.state_dict()
        t_r = 0.8
        g = collections.OrderedDict()
        for s in d.keys():
            g[s] = t_r*t[s] + (1-t_r)*d[s]
        self.target.load_state_dict(g)

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
        # total['sla2'] = []
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
        i = 0
        for mc in cluster.machine_list:
            if not mc.matchTask(task):
                continue
            i += 1
            x = []
            mc_busytime = mc.busy_time
            x.append(len(cluster.task_buffer))
            x.append(total['busy_time_std'])
            x.append(total['busy_time_mean'])
            x.append(mc_busytime)
            ET = mc.tryExcuteTask(task)
            x.append(ET)


            # x.append(total['busy_price_std'])
            x.append(total['busy_price_mean'])

            x.append(ET*mc.price)
            # x.append(mc.taskExpend(task))
            MCFT = mc_busytime + ET + cluster.env.clock
            PFT = task.load_time + task.duration
            beyoud_time = max(MCFT-PFT, 0)
            # x.append(max(mc.busy_time - task.load_time + cluster.env.clock, 0))
            x.append(beyoud_time)

            # total['sla2'].append(max(mc.busy_time - task.load_time + cluster.env.clock, 0))
            total['sla'].append(beyoud_time)
            total['a_time'].append(ET)
            total['a_price'].append(ET*mc.price)
            X.append(x)
            choose_mc_list.append(mc)

        sss1 = ['a_time_mean', 'a_time_std', 'a_price_mean', 'a_price_std', 'sla_mean', 'sla_std']
        sss2 = ['a_time', 'a_price', 'sla'] #, 'sla2'
        for ssst in sss2:
            total[ssst+'_'+'mean'] = np.array(total[ssst]).mean()
            total[ssst+'_'+'std'] = np.array(total[ssst]).std()
            total[ssst + '_' + 'min'] = np.array(total[ssst]).min()
        for x in X:
            x.append(total['a_time_mean'])
            x.append(total['a_time_std'])
            x.append(total['a_price_mean'])
            x.append(total['a_price_std'])
            x.append(total['sla_mean'])
            x.append(total['sla_std'])
        #     x.append(i)

            # x.append(total['sla2_mean'])
            # x.append(total['sla2_std'])

            # x.append(total['sla2_min'])
            # x.append(total['a_time_min'])
            # x.append(total['sla_min'])
            # x.append(total['a_price_min'])

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

        import time
        # n1 = time.time()
        X, choose_mc_list = self.getS(cluster, index)
        # n2 = time.time()
        # print('get_S_time:', n2-n1)
        if len(X) == 0:
            return None
        # n1 = time.time()
        a_out = self.selectAction([X])
        # n2 = time.time()
        # print('selectA_time:', n2 - n1)
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
            t_s = time.strftime('%Y-%m-%d-%H-%M-%S', self.localtime)
            file_name = '{}-{}.pkl'.format(t_s, self.dqn.__class__.__name__)
            path = '{}/{}/{}/{}'.format(Parameters.SavePath, self.ModelSaveName, t_s, file_name)
            torch.save(self.dqn.state_dict(), path)
        pass

    def loadModel(self, filename=None, path=None):
        if path == None:
            path = '{}/{}'.format(Parameters.SavePath, filename)
            self.dqn.load_state_dict(torch.load(path))
            self.updataTarget()
        else:
            self.dqn.load_state_dict(torch.load(path))
            self.updataTarget()

    def getFilenamePrefix(self):
        t_s = time.strftime('%Y-%m-%d-%H-%M-%S', self.localtime)
        return {
            'file_name': '{}-{}.pkl'.format(t_s, self.dqn.__class__.__name__),
            'file_folder': self.dqn.__class__.__name__,
        }
        pass
    def getTimeStr(self):
        return time.strftime('%Y-%m-%d-%H-%M-%S', self.localtime)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dqn = DDQNOne(15, 1)
    a = dqn.input_layer.parameters()
    b = list(a)[0]
    a2 = dqn.hidden_layers[0].parameters()
    b2 = list(a2)[0]
    print('gg')
    pass