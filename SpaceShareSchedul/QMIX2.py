import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import collections
from Base.Parameters import Parameters, DRLParameters
import time
import numpy as np
import torch.optim as optim


class DRQN(nn.Module):
    def __init__(self, conf, device = None):
        super(DRQN, self).__init__()
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.conf = conf
        fact = 30
        input_dim = self.conf.obs_shape
        output_dim = self.conf.output_shape
        self.input_layer = nn.Linear(input_dim, fact * input_dim)  # .to(device)
        self.hidden_layers = nn.ModuleList()
        for i in range(2, 4):
            self.hidden_layers.append(nn.Linear(int(fact * input_dim / (i - 1)), int(fact * input_dim / i)))
        # self.hidden_layers.to(device)
        self.output_layer = nn.Linear(int(fact * input_dim / i), output_dim)  # .to(device)
        self.to(device)

    def forward(self, obs):
        #根据输入判定是否需要tensor换与to
        obs = obs.to(self.device)
        # obs = obs.reshape(-1, self.conf.obs_shape)
        q = self.forward_item(obs)

        return q

    def forward_item(self, x):
        '''
        前向传播
        :param x:
        :return:
        '''
        x = F.leaky_relu(self.input_layer(x))
        for h_l in self.hidden_layers:
            x = F.leaky_relu(h_l(x))
        x = self.output_layer(x)
        return x

    def init_weights(self, m):
    	# 对于全连接层
        if type(m) == nn.Linear:
        	# 利用正态分布的方法来初始化参数
            torch.nn.init.kaiming_normal_(m.weight)
            # m.bias.data.fill_将bias直接初始化为0.0（float类型）
            m.bias.data.fill_(0.0)

class QMIXNET(nn.Module):
    def __init__(self, conf):
        super(QMIXNET, self).__init__()
        """
        生成的hyper_w1需要是一个矩阵，但是torch NN的输出只能是向量；
        因此先生成一个（行*列）的向量，再reshape
        """
        # print(conf.state_shape)
        self.conf = conf
        if self.conf.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim,
                                                    self.conf.n_agents * self.conf.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim, self.conf.qmix_hidden_dim))
        else:
            # self.hyper_w1 = nn.Linear(self.conf.state_shape, self.conf.n_agents * self.conf.qmix_hidden_dim)
            self.hyper_w1 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.conf.qmix_hidden_dim, 1))

    # input: (batch_size, n_agents, qmix_hidden_dim)
    # q_values: (episode_num, max_episode_len, n_agents)
    # states shape: (episode_num, max_episode_len, state_shape)
    def forward(self, q_values, states):
        episode_num = q_values.size(0)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, 1, self.conf.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.conf.qmix_hidden_dim)
        q_values2 = q_values.view(-1, 1, 1)
        hidden = F.elu(torch.bmm(q_values2, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.conf.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)

        return q_total

class QMIX(object):
    def __init__(self, conf, device=None):
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.conf = conf

        self.eval_drqn_net = DRQN( self.conf).to(self.device)
        self.target_drqn_net = DRQN(self.conf).to(self.device)

        self.eval_qmix_net = QMIXNET(self.conf).to(self.device)
        self.target_qmix_net = QMIXNET(self.conf).to(self.device)
        self.training = True

    def updataTarget(self):
        self.updataTargetFromEval(self.eval_drqn_net, self.target_drqn_net)
        self.updataTargetFromEval(self.eval_qmix_net, self.target_qmix_net)

    def updataTargetFromEval(self, eval, target, t_r=0.8):
        d = eval.state_dict()
        t = target.state_dict()
        g = collections.OrderedDict()
        for s in d.keys():
            g[s] = t_r * t[s] + (1 - t_r) * d[s]
        target.load_state_dict(g)

    def parameters(self):
        p = list(self.eval_qmix_net.parameters()) + list(self.eval_drqn_net.parameters())
        return p

class QMIXMemory(object):
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
        o, u, s, a, r, o_, u_, s_, done = [], [], [], [], [], [], [], [], []
        result = (o, u, s, a, r, o_, u_, s_, done)
        for item in mini_batch:
            for o, o1 in zip(result, item):
                o.append(o1)
        return result

    def handle_track(self, tracks):
        l = len(tracks)
        if l < 2:
            return False

        o = tracks[l-1]['s'][0]
        # h = tracks[l-1]['s'][1]
        u = tracks[l-1]['s'][1]
        s = tracks[l-1]['s'][2]
        a = tracks[l-1]['a']
        r = tracks[l-1]['r']
        o_ = tracks[l-1]['s'][0]
        u_ = tracks[l-1]['s'][1]
        s_ = tracks[l-1]['s'][2]
        done = 1
        self.push((o, u, s, a, r, o_, u_, s_, done))
        i = l - 2
        #
        while i >= 0:
            o = tracks[i]['s'][0]
            # h = tracks[i]['s'][1]
            u = tracks[i]['s'][1]
            s = tracks[i]['s'][2]
            a = tracks[i]['a']
            r = tracks[i]['r']
            o_ = tracks[i+1]['s'][0]
            u_ = tracks[i+1]['s'][1]
            s_ = tracks[i+1]['s'][2]
            done = 0
            self.push((o, u, s, a, r, o_, u_, s_, done))
            i -= 1
        return True

    def buffSize(self):
        # 返回池子的大小
        return len(self.buffer)

    def clear(self):
        return self.buffer.clear()

class Schedule(object):
    ModelSaveName = 'QMIX'
    def __init__(self, device=None):
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.conf = QMIXConf()
        self.qmix = QMIX(self.conf,device)


        self._sample_limit = (0.1, 1)
        self.sample_decay = (self._sample_limit[1] - self._sample_limit[0]) / int(5000 / DRLParameters.train_step)
        self.sample_rand_rate = self._sample_limit[1]

        self.training = True
        self.localtime = time.localtime()

        self.eval_parameters = self.qmix.parameters()
        self.optimizer = optim.RMSprop(self.eval_parameters, lr=self.conf.learning_rate)
        pass

    def getQValues(self, sample):
        # sample_shape = sample.shape
        # o, h, u, s, a, r, o_, u_, s_, done = [None] * 10
        # for k, v in zip([o, h, u, s, a, r, o_, u_, s_, done], sample):
        #     k = torch.tensor(v, dtype=torch.float32)

        o, u, s, a, r, o_, u_, s_, done = sample
        o = torch.tensor(o, dtype=torch.float32)
        u = torch.tensor(u, dtype=torch.float32)
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        o_ = torch.tensor(o_, dtype=torch.float32)
        u_ = torch.tensor(u_, dtype=torch.float32)
        s_ = torch.tensor(s_, dtype=torch.float32)
        done_ = torch.tensor(done, dtype=torch.float32)

        # o = torch.tensor(o, dtype=torch.float32)

        o_shape = o.shape
        # h_shape = h.shape
        o1 = o.view(-1,o_shape[-1])
        # h1 = h.view(-1, h_shape[-1])
        q, qh =self.qmix.eval_drqn_net(o1)
        q2 = q.view(*o_shape[:-1])
        qt, qht = self.qmix.target_drqn_net(o1)
        qt2 = qt.view(*o_shape[:-1])
        return q2, qt2

    def train(self, memory):
        if self.sample_rand_rate > self._sample_limit[0]:
            self.sample_rand_rate -= (self.sample_decay*5)
            print(self.sample_rand_rate)
        gamma = DRLParameters.gamma
        optimizer = self.optimizer
        all_loss = 0
        recycle = 25
        for i in range(recycle):
            sample_result = memory.sample(DRLParameters.batch_size)
            o, u, s, a, r, o_, u_, s_, done = sample_result
            q_evals, q_targets = self.getQValues(sample_result)
            a = torch.tensor(a).to(self.device)
            a = a.view(*a.shape, -1)
            q_a_evals = torch.gather(q_evals, dim=1, index=a)
            u = torch.tensor(u).to(self.device)
            q_targets[u == 0] = -float('inf')
            q_targets_max = q_targets.max(dim=1)[0]

            s = torch.tensor(s, dtype=torch.float32).to(self.device)
            q_total_eval = self.qmix.eval_qmix_net(q_a_evals, s)
            q_total_target = self.qmix.target_qmix_net(q_targets_max, s)

            r = torch.tensor(r, dtype=torch.float32).view(-1,1,1).to(self.device)
            done = torch.tensor(done).view(-1,1,1).to(self.device)
            targets = r + self.conf.gamma * q_total_target * (1 - done)
            td_error = (q_total_eval - targets.detach())
            loss = (td_error ** 2).sum() / len(td_error)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
            optimizer.step()
            all_loss += loss.cpu().detach().numpy()

            # del a
            # del u
            # del s
            # del done, r
            # del q_evals, q_a_evals, q_targets_max, targets
            # del td_error, loss
            # del q_targets, q_total_eval, q_total_target
            self.qmix.eval_hidden[0] = self.qmix.eval_hidden[0].detach()
        torch.cuda.empty_cache()
        return all_loss / recycle

    def updataTarget(self):
        self.qmix.updataTarget()

    def getS(self, cluster, index=0):
        task = cluster.task_buffer[index]

        X = []
        choose_mc_list = []
        available_mc = []

        #S
        total = {}
        S = []
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
        S.append(total['busy_time_std'])
        S.append(total['busy_time_mean'])
        S.append(total['busy_price_std'])
        S.append(total['busy_price_mean'])


        for mc in cluster.machine_list:
            # if not mc.matchTask(task):
            #     continue
            x = []
            # x.append(task.mips)
            # x.append(task.bw)
            # x.append(task.mips_length)
            # x.append(task.bw_length)
            mc_busytime = mc.busy_time
            x.append(mc_busytime)
            ET = mc.tryExcuteTask(task)
            x.append(ET)
            x.append(ET * mc.price)
            # x.append(mc.mips)
            # x.append(mc.bw)
            # x.append(mc.price)
            if mc.matchTask(task):
                ET = mc.tryExcuteTask(task)
                MCFT = mc_busytime + ET + cluster.env.clock
                PFT = task.load_time + task.duration
                beyoud_time = max(MCFT - PFT, 0)
                x.append(beyoud_time)
                available_mc.append(1)

                total['sla'].append(beyoud_time)
                total['a_time'].append(ET)
                total['a_price'].append(ET * mc.price)
            else:
                x.append(0)
                available_mc.append(0)
            choose_mc_list.append(mc)
            X.append(x)


        sss2 = ['a_time', 'a_price', 'sla']  # , 'sla2'
        for ssst in sss2:
            total[ssst + '_' + 'mean'] = np.array(total[ssst]).mean()
            total[ssst + '_' + 'std'] = np.array(total[ssst]).std()
            total[ssst + '_' + 'min'] = np.array(total[ssst]).min()

        S.append(total['a_time_mean'])
        S.append(total['a_time_std'])
        S.append(total['a_price_mean'])
        S.append(total['a_price_std'])
        S.append(total['sla_mean'])
        S.append(total['sla_std'])
        return [X, None, available_mc, S], choose_mc_list

    def task2mc(self, cluster, index=0):
        if len(cluster.task_buffer) == 0:
            return None
        task = cluster.task_buffer[index]
        X, choose_mc_list = self.getS(cluster, index)
        a_out = self.selectAction(X)
        try:
            return {choose_mc_list[a_out]:[{
                'task':task,
                's':X,
                'a':a_out,
            },]}
        except IndexError as e:
            print(e)
        pass

    def selectAction(self, X):
        randv = random.random()
        training = self.training
        X0 = torch.tensor(X[0], dtype=torch.float32)
        X1 = X[1]
        u = X[2]
        available_mc = X[2]
        a = self.qmix.eval_drqn_net(X0)
        available_mc = np.array(available_mc)
        a[available_mc == 0][1] = -float('inf')
        # self.qmix.eval_hidden[0] = h.detach()
        if training:
            # a = self.dqn(x)
            # return Universal.sample(a[0].detach().numpy())
            if randv > self.sample_rand_rate:
                # 按模型来
                # a = self.qmix.eval_drqn_net(X[0], X[1])
                return a[0].argmax().item()
            else:
                # 随机
                i = 0
                t = 0
                s = np.random.randint(1, 1+np.sum(u))
                for u_t in u:
                    t += u_t
                    if t == s:
                        break
                    i += 1
                a = i
                return a
        else:
            # a = self.qmix.eval_drqn_net(X[0], X[1])
            return a[0].argmax().item()

    def saveModelWithModel(self, model, path=None):
        if path == None:
            t_s = time.strftime('%Y-%m-%d-%H-%M-%S', self.localtime)
            file_name = '{}-{}.pkl'.format(t_s, model.__class__.__name__)
            path = '{}/{}/{}/{}'.format(Parameters.SavePath, self.ModelSaveName, t_s, file_name)
            torch.save(model.state_dict(), path)

    def saveModel(self, path=None):
        if path == None:
            self.saveModelWithModel(self.qmix.eval_drqn_net)
            self.saveModelWithModel(self.qmix.eval_qmix_net)
            pass

    def loadModel(self, filename=None, path=None):
        if path == None:
            raise Exception('还没处理此情况')
            path = '{}/{}'.format(Parameters.SavePath, filename)
            self.dqn.load_state_dict(torch.load(path))
            self.updataTarget()
        else:
            self.qmix.eval_drqn_net.load_state_dict(torch.load(path))
            self.updataTarget()

    def getTimeStr(self):
        return time.strftime('%Y-%m-%d-%H-%M-%S', self.localtime)

class QMIXConf(object):
    learning_rate =1e-3
    grad_norm_clip = 10



    obs_shape = 9
    state_shape = 4
    output_shape = 2
    n_agents = 54
    two_hyper_layers = False
    hyper_hidden_dim = 64
    qmix_hidden_dim = 32
    fact = 30
    drqn_hidden_dim = fact * obs_shape # GRU hidden
    gamma = 0.95