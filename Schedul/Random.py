from Base.Parameters import Parameters
import numpy as np
import random
from Schedul import Reward

class RandomMcSchedule(object):

    def task2mc(self, cluster):
        total_a_source ={}
        for s in Parameters.BaseSource:
            total_a_source[s] = 0

        task_list = cluster.task_buffer[:]
        if len(task_list) == 0:
            return None

        task = cluster.task_buffer[0]

        # 找到可以支持此task 的 mc
        match_task_mc = []
        for t_mc in cluster.machine_list:
            if t_mc.matchTask(task):
                match_task_mc.append(t_mc)
        if len(match_task_mc) == 0:
            return None

        mc_index = random.randint(0, len(match_task_mc)-1)
        match_mc = match_task_mc[mc_index]

        return {match_mc: [{
            'task': task,
            's': None,
            'a': None,
        }, ]}

    def getReward(self, **kwargs):
        cluster = kwargs['cluster']
        task = kwargs['task']
        mc = kwargs['mc']
        wait = cluster.env.clock - task.load_time
        dut = mc.tryExcuteTask(task)
        ext = max(wait - task.duration + dut, 0)

        return Reward.reward4Cluster_Makespan(wait, dut, ext, task.duration)
        pass

class DoubleRandomSchedule(object):

    def task2mc(self, cluster):
        total_a_source ={}
        for s in Parameters.BaseSource:
            total_a_source[s] = 0

        task_list = cluster.task_buffer[:]
        if len(task_list) == 0:
            return None

        task_index = random.randint(0, len(task_list) - 1)
        task = task_list[task_index]

        # 找到可以支持此task 的 mc
        match_task_mc = []
        for t_mc in cluster.machine_list:
            if t_mc.matchTask(task):
                match_task_mc.append(t_mc)
        if len(match_task_mc) == 0:
            return None

        mc_index = random.randint(0, len(match_task_mc)-1)
        match_mc = match_task_mc[mc_index]

        return {match_mc: [{
            'task': task,
            's': None,
            'a': None,
        }, ]}