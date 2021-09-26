from Base.Parameters import Parameters
import numpy as np

class MaxMinMakespanSchedule(object):

    def task2mc(self, cluster):

        task_list = cluster.task_buffer[:]
        if len(task_list) == 0:
            return None


        max_task = cluster.task_buffer[0]
        for task in task_list:
            if task.duration > max_task.duration:
                max_task = task

        # 找到可以支持此task 的 mc
        min_time = None
        match_min_task_mc = []
        match_mc = None
        for t_mc in cluster.machine_list:
            if t_mc.matchTask(max_task):
                f_time = t_mc.busy_time + t_mc.tryExcuteTask(max_task)
                if min_time == None:
                    min_time = f_time
                    match_mc = t_mc
                elif f_time < min_time:
                        min_time = f_time
                        match_mc = t_mc
        if min_time == None:
            return None
        # 将任务分配给 匹配mc
        return {match_mc: [{
            'task': max_task,
            's': None,
            'a': None,
        }, ]}