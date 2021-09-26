# from Modules.Cluster import Cluster
from Base.Parameters import Parameters
class DRF(object):

    def task2mc(self, cluster):
        # cluster.machine_list
        # cluster.task_buffer
        # total_a_source = {
        #     'mips':0,
        #     'bw':0,
        # } # 集群可用资源
        total_a_source ={}
        for s in Parameters.BaseSource:
            total_a_source[s] = 0

        task_list = cluster.task_buffer[:]
        if len(task_list) == 0:
            return None
        # a_mc_source[mc] = {
        #     'mips': mc.a_mips,
        #     'bw': mc.a_bw,
        #     'task':[],
        # }
        a_mc_source ={}
        # 对集群资源进行统计 生成a_mc_source & total_a_source
        for mc in cluster.machine_list:
            a_mc_source[mc] = {}
            for s in Parameters.BaseSource:
                total_a_source[s] += mc.a_source[s]
                a_mc_source[mc][s] = mc.a_source[s]
            a_mc_source[mc]['task'] = []

        # 找出这批任务主导资源与任务
        min_task = None
        min_task_pd = None
        min_property = None
        for task in task_list:
            task_pd = 0
            task_property = None
            for s in Parameters.BaseSource:
                t_task_pd = task.need_source[s] / total_a_source[s]
                if t_task_pd > task_pd:
                    task_pd = t_task_pd
                    task_property = s
            if min_task_pd == None or min_task_pd > task_pd:
                min_task_pd = task_pd
                min_task = task
                min_property = task_property


        # 找到可以支持此task 的 mc
        match_min_task_mc = []
        for t_mc in cluster.machine_list:
            if t_mc.matchTask(min_task):
                match_min_task_mc.append(t_mc)
        # for t_mc, a_source in a_mc_source.items():
        #     mark = True
        #     for s in Parameters.BaseSource:
        #         if a_source[s] < min_task.need_source[s]:
        #             mark = False
        #             break
        #     if mark:
        #         match_min_task_mc.append(t_mc)
        if len(match_min_task_mc) == 0:
            return None
        # 在支持mc中找到 在可使之min mc
        max_property_val = 0
        match_mc = None
        for mc in match_min_task_mc:
            if mc.a_source[min_property]/mc.source[min_property] > max_property_val:
                match_mc = mc
                max_property_val = mc.a_source[min_property]/mc.source[min_property]
        # 将任务分配给 匹配mc

        return {match_mc: [{
            'task': min_task,
            's': None,
            'a': None,
        }, ]}


    def task2mc_s(self, cluster):
        # cluster.machine_list
        # cluster.task_buffer
        # total_a_source = {
        #     'mips':0,
        #     'bw':0,
        # } # 集群可用资源
        total_a_source ={}
        for s in Parameters.BaseSource:
            total_a_source[s] = 0

        task_list = cluster.task_buffer[:]
        # a_mc_source[mc] = {
        #     'mips': mc.a_mips,
        #     'bw': mc.a_bw,
        #     'task':[],
        # }
        a_mc_source ={}
        # 对集群资源进行统计 生成a_mc_source & total_a_source
        for mc in cluster.machine_list:
            a_mc_source[mc] = {}
            for s in Parameters.BaseSource:
                total_a_source[s] += mc.a_source[s]
                a_mc_source[mc][s] = mc.a_source[s]
            a_mc_source[mc]['task'] = []

        while len(task_list) > 0:
            # 找出这批任务主导资源与任务
            max_task = None
            max_task_pd = 0
            max_property = None
            min_task = None
            min_task_pd = None
            min_property = None
            for task in task_list:
                task_pd = 0
                task_property = None
                for s in Parameters.BaseSource:
                    t_task_pd = task.need_source[s] / total_a_source[s]
                    if t_task_pd > task_pd:
                        task_pd = t_task_pd
                        task_property = s
                if min_task_pd == None or min_task_pd > task_pd:
                    min_task_pd = task_pd
                    min_task = task
                    min_property = task_property


            # 找到可以支持此task 的 mc
            match_min_task_mc = []
            for t_mc, a_source in a_mc_source.items():
                mark = True
                for s in Parameters.BaseSource:
                    if a_source[s] < min_task.need_source[s]:
                        mark = False
                        break
                if mark:
                    match_min_task_mc.append(t_mc)
            if len(match_min_task_mc) == 0:
                break;
            # 在支持mc中找到 在可使之min mc
            max_property_val = 0
            match_mc = None
            for mc in match_min_task_mc:
                if a_mc_source[mc][min_property] > max_property_val:
                    match_mc = mc
                    max_property_val = a_mc_source[match_mc][min_property]
            # 将任务分配给 匹配mc
            for s in Parameters.BaseSource:
                a_mc_source[mc][s] -= min_task.need_source[s]
                total_a_source[s] -= min_task.need_source[s]
            a_mc_source[mc]['task'].append(min_task)
            task_list.remove(min_task)

        return a_mc_source


    def extask2mc(self, cluster):
        total_a_source = {}
        for s in Parameters.BaseSource:
            total_a_source[s] = 0
        clock = cluster.env.clock
        task_list = cluster.task_buffer[:]
        # a_mc_source[mc] = {
        #     'mips': mc.a_mips,
        #     'bw': mc.a_bw,
        #     'task':[],
        # }
        a_mc_source = {}
        # 对集群资源进行统计 生成a_mc_source & total_a_source
        for mc in cluster.machine_list:
            a_mc_source[mc] = {}
            for s in Parameters.BaseSource:
                total_a_source[s] += mc.a_source[s]
                a_mc_source[mc][s] = mc.a_source[s]
            a_mc_source[mc]['task'] = []

        while len(task_list) > 0:
            # 找出这批任务主导资源与任务
            min_task = None
            min_task_pd = None
            min_property = None
            for task in task_list:
                task_pd = 0
                task_property = None
                for s in Parameters.BaseSource:
                    t_task_pd = task.ex_need_source(clock)[s] / total_a_source[s]
                    if t_task_pd > task_pd:
                        task_pd = t_task_pd
                        task_property = s
                if min_task_pd == None or min_task_pd > task_pd:
                    min_task_pd = task_pd
                    min_task = task
                    min_property = task_property

            # 找到可以支持此task 的 mc
            match_min_task_mc = []
            for t_mc, a_source in a_mc_source.items():
                mark = True
                for s in Parameters.BaseSource:
                    if a_source[s] < min_task.ex_need_source(clock)[s]:
                        mark = False
                        break
                if mark:
                    match_min_task_mc.append(t_mc)
            if len(match_min_task_mc) == 0:
                break;
            # 在支持mc中找到 在可使之min mc
            max_property_val = 0
            match_mc = None
            for mc in match_min_task_mc:
                if a_mc_source[mc][min_property] > max_property_val:
                    match_mc = mc
                    max_property_val = a_mc_source[match_mc][min_property]
            # 将任务分配给 匹配mc
            for s in Parameters.BaseSource:
                a_mc_source[match_mc][s] -= min_task.ex_need_source(clock)[s]
                total_a_source[s] -= min_task.ex_need_source(clock)[s]
            a_mc_source[match_mc]['task'].append(min_task)
            task_list.remove(min_task)

        return a_mc_source