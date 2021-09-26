from Base.SimBaseInterfaces import BaseObj
from Base.Parameters import Parameters
from Modules.Cluster import Cluster

from random import Random

class DataCenter(BaseObj):
    cluster_list = None
    env = None
    task_list = None
    def __init__(self, env, config=None, mc_schedule=None, clu_schedule=None):
        BaseObj.__init__(self, env)
        self.env = env
        self.cluster_list = []
        self.task_list = []
        self.mc_schedule = mc_schedule
        self.clu_schedule = clu_schedule
        if config:
            self.loadConfig(config)

    def init(self):
        pass

    def addTask(self, tasks):
        '''
        将任务加入缓存
        可以考虑增加排序
        :param tasks:
        :return:
        '''
        self.task_list = self.task_list + tasks

    def allocationTask2Cluster(self):
        clu_l = len(self.cluster_list)

        r = Random()

        pre_allocat_map = {i:[] for i in self.cluster_list}
        while len(self.task_list) >0 :
            task = self.task_list.pop()
            index = r.randint(0,clu_l-1)
            pre_allocat_map[self.cluster_list[index]].append(task)

        for cluster in pre_allocat_map.keys():
            if len(pre_allocat_map[cluster]) > 0:
                self.send(cluster.id, data={
                    'event_type':Parameters.ClusterType.TASK_ARRIVE,
                    'tasks':pre_allocat_map[cluster],
                })

        pass

    def process_event(self, event):
        data = event['data']
        event_type = data['event_type']
        if event_type == Parameters.DataCenterType.SEARCH_DC:
            self.send(event['src'],data={
                'event_type':Parameters.DataCenterType.REPORT_SEARCH_DC,
                'dc_id':self.id,
            })
            return None
        if event_type == Parameters.DataCenterType.TASK_ARRIVE:
            '''
            将任务压入自身缓存，并判定是否触发调度
            '''
            self.addTask(data['task_list'])


            self.allocationTask2Cluster()



            pass

    def loadConfig(self, config= None):
        if config == None:
            config = Parameters.DC_config
        try:
            clusters_conf = config['Clusters']
        except TypeError as e:
            print(e)
        for cluster_c in clusters_conf:
            self.cluster_list.append(Cluster(self.env, cluster_c, mc_schedule=self.mc_schedule))
    pass