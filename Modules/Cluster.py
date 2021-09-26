from Base.SimBaseInterfaces import BaseObj
from Modules.Machine import Machine
from Base.Parameters import Parameters
from random import Random
from Schedul import Reward
import numpy as np



# from Schedul.DDQN import Schedule
class Cluster(BaseObj):
    '''
    任务顺序执行 即1:n

    '''
    task_buffer = None
    _all_mips = None
    _all_bw = None
    machine_list = None
    drl_track = None


    def __init__(self, env, config, mc_schedule):
        BaseObj.__init__(self, env)
        self.task_buffer = []
        self.machine_list = []
        self.load_config(config)
        self.schedul = mc_schedule
        self.drl_track = []
        self.recode_list =[]
        self.reward_list = []
        self.other_choose = []

        self.mean_source = None
        self.std_source = None

    def load_config(self, config):
        Machines_conf = config['Machines']
        for machine_c in Machines_conf:
            mc = Machine(machine_c, self)
            mc.id = self.id*Parameters.dc_property_cluster_mc_num_max + len(self.machine_list)
            self.machine_list.append(mc)

    def addTask(self, tasks:list):
        self.task_buffer = self.task_buffer + tasks

    def recodeInfo(self):
        mc_num = 0
        for mc in self.machine_list:
            mc_num += len(mc.task_list)
        item = {
            'time': self.env.clock,
            'balance': Reward.reward4Cluster_LoadBalance(self),
            'mc_num': mc_num,
        }
        self.recode_list.append(item)



    def schedulTask2Machine(self):
        if not Parameters.EXState:
            result = {
                'start_time': self.env.clock,
                'task_list':[],
                'r': None
            }
            while len(self.task_buffer) > 0:
                mc_infos_map = self.schedul.task2mc(self)
                try:
                    for mc, infos in mc_infos_map.items():
                        for data in infos:
                            task = data['task']
                            s = data['s']
                            a = data['a']
                            a_value = data.get('av', None)
                            # Reward
                            dur = mc.executeTask(task)
                            result['task_list'].append(task)
                            track_item = {
                                's': s,
                                'a': a,
                                'r': None,
                                'start': self.env.clock,
                                'dur': dur,
                                'diff_r': 0,
                                'err': False,
                                'a_value': a_value,
                                'task':task,
                                'task_recode':result,
                            }
                            self.task_buffer.remove(task)
                            self.drl_track.append(track_item)
                            if dur == -1:
                                continue
                                pass
                            if dur > 0:
                                self.send(self.id, data={
                                    'event_type': Parameters.MachineType.TASK_OVER,
                                    'mc': mc,
                                    'task': task,
                                }, dest=self.env.clock + dur)
                except AttributeError as e:
                    print(e)
            return True

    def init(self):
        pass

    def process_event(self, event):
        data = event['data']
        event_type = data['event_type']
        if event_type == Parameters.ClusterType.TASK_ARRIVE:
            self.addTask(data['tasks'])
            self.schedulTask2Machine()
            self.recodeInfo()
            return
        if event_type == Parameters.MachineType.TASK_OVER:
            mc = data['mc']
            task = data['task']
            mc.finishTask(task)
            '''
            并检查mc的任务缓存
            '''
            if len(mc.task_list) > 0:
                dur = mc.continueExeTask()
                task = mc.task_list[0]
                self.send(self.id, data={
                    'event_type': Parameters.MachineType.TASK_OVER,
                    'mc': mc,
                    'task': task,
                }, dest=self.env.clock + dur)
            self.recodeInfo()
        pass

    def tryMeanExcuteTask(self, task):
        dur = 0
        dur += task.mips_length/self.source_mean_std['mean']['mips']
        dur += task.bw_length/self.source_mean_std['mean']['bw']
        return dur

    # def getMaxTask(self):
    #     task = None
    #     if len(self.task_buffer) > 0:
    #         task = self.task_buffer[0]
    #         for t in self.task_buffer:
    #             if task.duration < t.duration:
    #                 task = t
    #     return task

    @property
    def source_list(self):
        total_source = {}
        for s in Parameters.BaseSource:
            total_source[s] = []

        for mc in self.machine_list:
            for s in Parameters.BaseSource:
                total_source[s].append(mc.source[s])
        return total_source

    @property
    def source_mean_std(self):
        source_list = self.source_list
        if self.mean_source is None:
            self.mean_source = {}
            self.std_source = {}
            for s in Parameters.BaseSource:
                np_s = np.array(source_list[s])
                self.mean_source[s] = np_s.mean()
                self.std_source[s] = np_s.std()
            for s in Parameters.OtherSource:
                np_s = np.array(source_list[s])
                self.mean_source[s] = np_s.mean()
                self.std_source[s] = np_s.std()
        return {
            'mean': self.mean_source,
            'std': self.std_source,
        }

    @property
    def total_task_request(self):
        total_request = {}
        for s in Parameters.BaseSourceLength:
            total_request[s] = []
        for task in self.task_buffer:
            for s in Parameters.BaseSourceLength:
                total_request[s].append(task.need_source[s])
        return total_request

    @property
    def mc_busy_time_list(self):
        result = []
        for mc in self.machine_list:
            result.append(mc.busy_time)
        return np.array(result)

    @property
    def busy_time(self):
        return self.mc_busy_time_list.max() + self.mean_wait_task_time

    @property
    def mean_wait_task_time(self):
        '''
        平均任务量/平均计算能力
        反映平均等待时间？
        这不合适,所以*上了任务数整除vm数
        :return:
        '''
        total_source = self.source_list
        total_request = self.total_task_request

        mwt = 0
        for s, sl in zip(Parameters.BaseSource, Parameters.BaseSourceLength):
            m = np.array(total_source[s]).mean()
            ml = np.array(total_request[sl]).mean()
            mwt += (ml/m)
        mwt *= (1 + len(self.task_buffer) // (len(self.machine_list) +1))
        return mwt