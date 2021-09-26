from Base.Parameters import Parameters
import numpy as np

class MinMakespanSchedule(object):

    def task2mc(self, cluster):
        total_a_source ={}
        for s in Parameters.BaseSource:
            total_a_source[s] = 0

        task_list = cluster.task_buffer[:]
        if len(task_list) == 0:
            return None

        a_mc_source ={}
        # 对集群资源进行统计 生成a_mc_source & total_a_source
        for mc in cluster.machine_list:
            a_mc_source[mc] = {}
            for s in Parameters.BaseSource:
                total_a_source[s] += mc.a_source[s]
                a_mc_source[mc][s] = mc.a_source[s]
            a_mc_source[mc]['task'] = []

        min_task = cluster.task_buffer[0]

        # 找到可以支持此task 的 mc
        min_time = None
        match_min_task_mc = []
        match_mc = None
        for t_mc in cluster.machine_list:
            if t_mc.matchTask(min_task):
                f_time = t_mc.busy_time + t_mc.tryExcuteTask(min_task)
                # f_time /= 20
                # f_price = t_mc.taskExpend(min_task)
                # f_price /= 50
                # f_r = Parameters.r1_rate*f_time + Parameters.r2_rate*f_price
                # f_time = f_r
                if min_time == None:
                    min_time = f_time
                    match_mc = t_mc
                elif f_time < min_time:
                        min_time = f_time
                        match_mc = t_mc
        if min_time == None:
            return None


        # # 在支持mc中找到 在可使之min mc
        # def balance():
        #     total_source_p = {}  # 每台机器对应的可用资源百分比
        #     for s in Parameters.BaseSource:
        #         total_source_p[s + '_p'] = []
        #     for mc, info in a_mc_source.items():
        #         for s in Parameters.BaseSource:
        #             total_source_p[s + '_p'].append(info[s]/mc.source[s])
        #     result = 0
        #     for s in Parameters.BaseSource:
        #         total_source_p[s + '_std'] = np.array(total_source_p[s + '_p']).std()  # 集群内各项资源对应的标准差
        #         total_source_p[s + '_mean'] = np.array(total_source_p[s + '_p']).mean()
        #         result += total_source_p[s + '_std']
        #     return result


        # min_balance = None
        # match_mc = None
        # for mc in match_min_task_mc:
        #     for s in Parameters.BaseSource:
        #         a_mc_source[mc][s] -= min_task.need_source[s]
        #     score = balance()
        #     if min_balance==None or score < min_balance:
        #         min_balance = score
        #         match_mc = mc
        #     for s in Parameters.BaseSource:
        #         a_mc_source[mc][s] += min_task.need_source[s]
        # 将任务分配给 匹配mc
        return {match_mc: [{
            'task': min_task,
            's': None,
            'a': None,
        }, ]}