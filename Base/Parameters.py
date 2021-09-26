from enum import Enum
import random


class Parameters(object):

    MipsScale = 1000 # 将计算机的mips
    RuntimeScale = 1000/MipsScale
    FileLengthScale = 1000000
    EPIS = 0.5
    EXState = False
    ROOTPath= 'E:/Storage/Code/Code/li-Cloudsim210805-first-final/'
    SavePath= 'E:/Storage/Code/Code/li-Cloudsim210805-first-final/ModelSave/'
    BaseSource = ('mips', 'bw')
    BaseSourceLength = ('mips_length', 'bw_length')
    OtherSource = ('price', )
    SCOREResult = ('total_r', 'total_gae_r', 'task_num', 'loadBalance', 'end_time', 'expend_score', 'sla_mean', 'sla_std', 'response_time_mean', 'response_time_srd', 'response_time_std','task_time_score', 'r1_score')
    BaseSourceLimit={
        'mips':(0,5000),
        'bw':(0, 500),
    }
    BrokerEventType = Enum('BrokerEvnetType', (
        'EXECUTE_JOB',
        'EXECUTE_TASK_OVER',
        'ALLOCATE_TASK',
        'WAIT_REPORT'
    ))
    DataCenterType = Enum('DataCenterType',(
        'TASK_ARRIVE', #任务到达数据中心
        'SEARCH_DC', # 是不是DC，是的话回个话
        'REPORT_SEARCH_DC', # 回报自身信息
    ))
    ClusterType = Enum('ClusterType',(
        'TASK_ARRIVE',
    ))
    MachineType = Enum('MachineType',(
        'EXECUTE_TASK',
        'TASK_OVER',
    ))
    SimEventType = Enum("EventType", ("SEND", 'BROADCAST'))
    SimSTATEType = Enum('STATEType', ('INIT', 'STOP', 'RUN', 'PAUSE'))
    VMState = Enum('VMState', ('RUN', 'STOP', 'TIMESHARE', 'SPACESHARE'))

    ShareModel = VMState.SPACESHARE
    dc_property_cluster_mc_num_max = 100
    r1_rate = 0.2
    r2_rate = 0.8

    MS1= [
        {
            'mips': 5000,
            'bw': 300,
        },
        {
            'mips': 2000,
            'bw': 50,
        },
        {
            'mips': 2000,
            'bw': 100,
        },
        {
            'mips': 3000,
            'bw': 100,
        },
        {
            'mips': 3000,
            'bw': 100,
        },
        {
            'mips': 4000,
            'bw': 200,
        },
        {
            'mips': 5000,
            'bw': 200,
        },
        {
            'mips': 6000,
            'bw': 300,
        },
        {
            'mips': 5000,
            'bw': 300,
        },
        {
            'mips': 4000,
            'bw': 200,
        },
        {
            'mips': 7000,
            'bw': 300,
        },
        {
            'mips': 3000,
            'bw': 100,
        },
        {
            'mips': 3000,
            'bw': 100,
        },
        {
            'mips': 4000,
            'bw': 200,
        },
        {
            'mips': 5000,
            'bw': 200,
        },
        {
            'mips': 6000,
            'bw': 300,
        },
        {
            'mips': 5000,
            'bw': 300,
        },
        {
            'mips': 4000,
            'bw': 200,
        },
    ]

    # MS1 = [
    #     {
    #         'mips': 1000,
    #         'bw': 300,
    #     },
    #     {
    #         'mips': 5000,
    #         'bw': 50,
    #     },
    #     {
    #         'mips': 5000,
    #         'bw': 100,
    #     },
    #     {
    #         'mips': 3000,
    #         'bw': 100,
    #     },
    #     {
    #         'mips': 3000,
    #         'bw': 200,
    #     },
    #     {
    #         'mips': 4000,
    #         'bw': 200,
    #     },
    #     {
    #         'mips': 5000,
    #         'bw': 200,
    #     },
    #     {
    #         'mips': 2000,
    #         'bw': 300,
    #     },
    #     {
    #         'mips': 5000,
    #         'bw': 300,
    #     },
    #     {
    #         'mips': 4000,
    #         'bw': 200,
    #     },
    #     {
    #         'mips': 7000,
    #         'bw': 300,
    #     },
    #     {
    #         'mips': 3000,
    #         'bw': 100,
    #     },
    #     {
    #         'mips': 3000,
    #         'bw': 100,
    #     },
    #     {
    #         'mips': 4000,
    #         'bw': 200,
    #     },
    #     {
    #         'mips': 5000,
    #         'bw': 200,
    #     },
    #     {
    #         'mips': 5000,
    #         'bw': 300,
    #     },
    #     {
    #         'mips': 3000,
    #         'bw': 300,
    #     },
    #     {
    #         'mips': 4000,
    #         'bw': 200,
    #     },
    # ]

    MS1 += (MS1*2)
    DC_config = {
        'Clusters' : [
            {
                'Machines' : MS1[:]
            },
        ],
    }


    mips_price = 0.00007
    stair_bw = [200]
    bw_price = [0.063, 0.248]


    # @staticmethod
    # def calculate_cost(mc=None, task=None):
    #     x_mips = None
    #     x_bw = None
    #     price = None
    #     if x_bw < Parameters.stair_bw:
    #         price = x_bw * Parameters.bw_price[0] + x_mips * Parameters.mips_price
    #     else:
    #         price = Parameters.stair_bw * Parameters.bw_price[0] + (x_bw - Parameters.stair_bw) * Parameters.bw_price[1] + x_mips * Parameters.mips_price
    #     return price



class DRLParameters(object):
    memory_size = 1*10**3 #5
    batch_size = 50
    epochs = 10000
    train_step = 3 #几个回合 进行一次学习
    buffer_mini_size = batch_size*5
    gamma = 0.9 #0.99 #累计回报
    learning_rate = 0.00001 # 学习率
    updata_target_step = train_step * 3

    mark = 0


class TaskGenerateParam:
    '''
    以秒为基础时间单位
    '''
    lam = 3  # 平均每秒多少批任务到达
    E1 = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    E2 = [1, 2, 41, 42, 43, 7, 8, 50, 51, 52, 53, 54, 0]
    # E = E[:2]
    # E = [1,2,3,4,6,7,8,9,9,29,0, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    E5 = [1, 2, 3, 4, 6, 7, 8, 9, 9, 29, 0, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 2, 3, 4, 6, 7, 8, 9, 9, 29]
    E = E5
    arrive_time_seeds = E
    task_num_seeds = E
    task_param_seeds = E
    task_num_limit = (2,6) # 2~9 = 7+2   *5   *8
    task_num_max = 100 # task_num_limit[1] < task_num_max
    # task_time_limit = (10+5,20+10) # +5  +10

    task_mips_limit = (100,5000) # 4000
    task_bw_limit = (40,250) # 200
    task_duration_limit = (15/3,30)
    task_mips_seed = E
    task_bw_seed = E
    task_duration_seed = E
    # task_split_duration_seed = E

    def __init__(self):
        self.index = 0

        pass

    def get_param(self):
        index = self.index

        self.index += 1
        if self.index >= len(self.arrive_time_seeds):
            return None
        return {
            'time_seed':self.arrive_time_seeds[index],
            'task_num_seed':self.task_num_seeds[index],
            'mips_seed':self.task_mips_seed[index],
            'bw_seed':self.task_bw_seed[index],
            'duration_seed':self.task_duration_seed[index],
            # 'split_duration_seed':self.task_split_duration_seed[index],
        }
        # return {
        #     'time_seed': random.random(),
        #     'task_num_seed': random.random(),
        #     'mips_seed': random.random(),
        #     'bw_seed': random.random(),
        #     'duration_seed': random.random(),
        #     # 'split_duration_seed':self.task_split_duration_seed[index],
        # }


    @classmethod
    def withE(cls):
        E = cls.E
        cls.arrive_time_seeds = E
        cls.task_num_seeds = E
        cls.task_param_seeds = E

        cls.task_mips_seed = E
        cls.task_bw_seed = E
        cls.task_duration_seed = E
    pass


import json
import pickle

if __name__ == '__main__':
    fq = open('../ModelSave/1.PyClass', 'wb')
    pickle.dump(Parameters, fq, protocol = None, fix_imports = True )
    fq.close()
    fq = open('../ModelSave/1.PyClass', 'rb')
    asd = pickle.load(fq, encoding='bytes')
    fq.close()
    print('g')
    pass