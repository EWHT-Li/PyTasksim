from Base.Parameters import Parameters
import numpy as np

class RoundRobinSchedule(object):

    def __init__(self):
        self.index = 0

    def task2mc(self, cluster):

        task_list = cluster.task_buffer[:]
        if len(task_list) == 0 or len(cluster.machine_list) == 0:
            return None
        min_task = cluster.task_buffer[0]

        match_mc = None
        i = 0
        while i < len(cluster.machine_list):
            i + 1
            if cluster.machine_list[self.index].matchTask(min_task):
                match_mc = cluster.machine_list[self.index]
                self.index += 1
                self.index = self.index % len(cluster.machine_list)
                break
            self.index += 1
            self.index = self.index % len(cluster.machine_list)

        if i >= len(cluster.machine_list):
            raise Exception("不科学，不可能啊")

        # 将任务分配给 匹配mc
        return {match_mc: [{
            'task': min_task,
            's': None,
            'a': None,
        }, ]}