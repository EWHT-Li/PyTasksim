from Base.SimBaseInterfaces import BaseObj
from Base.Parameters import TaskGenerateParam, Parameters
from random import Random
from Modules.JobTask import Task
from Tool.Universal import Universal
import math

class Broker(BaseObj):

    def __init__(self, env, data_center=None):
        BaseObj.__init__(self, env)
        self.data_center = data_center
        self.task_param = TaskGenerateParam()
        self.r = Random()
        self.gen_index = 0
        self.dc_id_set = set()

    def init(self):
        '''

        群发，广播，问询所有DC，建立DC表
        '''

        self.broadSend(data={
            'event_type': Parameters.DataCenterType.SEARCH_DC,
        })
        self.send(self.id, data={
            'event_type': Parameters.BrokerEventType.ALLOCATE_TASK,
        },dest=self.env.clock+1)
        pass

    def taskGenerate(self):
        gen_index = self.gen_index
        self.gen_index += 1
        r = self.r
        task_param = self.task_param
        task_seed = task_param.get_param()
        if task_seed == None:
            return None, None
        time_seed = task_seed['time_seed']
        task_num_seed = task_seed['task_num_seed']
        mips_seed = task_seed['mips_seed']
        bw_seed = task_seed['bw_seed']
        duration_seed = task_seed['duration_seed']

        time_interval = -math.log(1 - Universal.getrand(time_seed)) / task_param.lam
        dest_time = self.env.clock + time_interval
        # send() # 通知自己到点发下一波任务，和将这波任务交给数据中心
        task_num = Universal.getrand(task_num_seed)
        tasks_count = int(self.task_param.task_num_limit[0]+ task_param.task_num_limit[1]*task_num)

        mips_seed = Universal.getrand(mips_seed) # seed_a ->seed_b
        bw_seed = Universal.getrand(bw_seed) # seed_a -> seed_b
        duration_seed = Universal.getrand(duration_seed) #

        task_list = []

        for i in range(tasks_count):
            t_mips = task_param.task_mips_limit[0] + mips_seed*task_param.task_mips_limit[1]
            t_bw = task_param.task_bw_limit[0] + bw_seed*task_param.task_bw_limit[1]
            t_duration = task_param.task_duration_limit[0] + duration_seed*task_param.task_duration_limit[1]
            t_mips_duration = t_duration*mips_seed/(mips_seed+bw_seed)
            t_bw_duration = t_duration*bw_seed/(mips_seed+bw_seed)

            tmp_task = Task() # mips bw
            tmp_task.mips = int(t_mips)
            tmp_task.bw = int(t_bw)
            tmp_task.duration = int(t_mips_duration) + int(t_bw_duration)
            try:
                1/tmp_task.duration
            except ZeroDivisionError as e:
                print(e)
            tmp_task.task_length = int(t_mips_duration) * int(t_mips)
            tmp_task.mips_length = tmp_task.task_length
            tmp_task.bw_length = int(t_bw_duration) * int(t_bw)
            tmp_task.task_id = task_param.task_num_max*gen_index + i
            tmp_task.load_time = self.env.clock

            mips_seed = Universal.getrand(mips_seed) # seed_b -> seed_c
            bw_seed = Universal.getrand(bw_seed) #
            duration_seed = Universal.getrand(duration_seed)

            task_list.append(tmp_task)
        return dest_time, task_list


    def allocate2DC(self, dest_time, task_list):
        dc_id_list = list(self.dc_id_set)
        self.send(dc_id_list[0], data={
            'event_type': Parameters.DataCenterType.TASK_ARRIVE,
            'task_list': task_list,
        })


    def process_event(self, event):
        data = event['data']
        event_type = data['event_type']
        if event_type == Parameters.DataCenterType.REPORT_SEARCH_DC:
            self.dc_id_set.add(data['dc_id'])
            return None
        if event_type == Parameters.BrokerEventType.ALLOCATE_TASK:
            dest_time, task_list = self.taskGenerate()
            if dest_time == None or len(task_list)==0:
                return None
            # if self.env.clock + dest_time == 1099276.952426154:
            #     print('gg')
            self.allocate2DC(dest_time, task_list)
            self.send(self.id, data={
                'event_type': Parameters.BrokerEventType.ALLOCATE_TASK,
            }, dest=dest_time)

            return None



    def run(self):
        '''
        将任务发送到对应数据中心
        根据参数计算下一批任务的时间，通知自己到点发送任务给数据中心
        '''
        pass


