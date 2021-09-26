from Base.Parameters import Parameters
import numpy as np
import random

class RandomSchedule(object):

    def task2mc(self, cluster):

        task_list = cluster.task_buffer[:]
        if len(task_list) == 0:
            return None


        min_task = cluster.task_buffer[0]

        # 找到可以支持此task 的 mc
        mc_list = []
        for t_mc in cluster.machine_list:
            if t_mc.matchTask(min_task):
                mc_list.append(t_mc)

        if len(mc_list) == 0:
            return None
        else:
            index = random.randint(0,len(mc_list)-1)
            match_mc = mc_list[index]

        # 将任务分配给 匹配mc
        return {match_mc: [{
            'task': min_task,
            's': None,
            'a': None,
        }, ]}