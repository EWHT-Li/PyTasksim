from Base.SimEnv import SimEnv
from Base.Parameters import Parameters, DRLParameters
from Modules.DataCenter import DataCenter
from Modules.Broker import Broker
from Base.SimBaseInterfaces import Base
from Modules.Cluster import Cluster
from Tool import Score
'''
PPO
'''

from Schedul.AC import SimpleA2CMemory
from Schedul.AC import Schedule
from Schedul.DRF import DRF
from Schedul.MinBalance import MinBalanceSchedule
from Schedul.Random import RandomMcSchedule, DoubleRandomSchedule
from Schedul import Reward
#
def evaluateSchedule(schdule):
    env = SimEnv()
    b = Base(env)
    dc = DataCenter(env, Parameters.DC_config, mc_schedule=schdule)
    broker = Broker(env)
    env.run()
    env.clock
    over_tasks = []
    for dc_id in broker.dc_id_set:
        t_dc = env.obj_list[dc_id]
        for t_clu in t_dc.cluster_list:
            Reward.reviseTrack(t_clu)
            for t_mc in t_clu.machine_list:
                over_tasks += t_mc.over_task_list
    abs_sla, comp_sla = Score.taskSLAScore(over_tasks)
    bl_score = Score.dcBalanceScore(dc)
    br,br1,br2,ext = Score.cluRewardScore(dc.cluster_list[0])
    other_br = Score.cluOtherRewardScore(dc.cluster_list[0])
    s = '{}:完成时间：{}；任务绝对SLA得分：均{}，标{}；任务相对SLA得分：均{}，标{}；均衡得分：{}'.format(schdule.__class__.__name__,
                                                                        env.clock, abs_sla.mean(), abs_sla.std(),
                                                                        comp_sla.mean(), comp_sla.std(), bl_score)
    if bl_score < 10:
        print(123)
    if bl_score >20 and bl_score < 21:
        Score.dcBalanceScore(dc)
        Score.cluRewardScore(dc.cluster_list[0])
        DRLParameters.mark = 1
    if DRLParameters.mark==1 and bl_score< 22 and bl_score>21:
        Score.dcBalanceScore(dc)
        Score.cluRewardScore(dc.cluster_list[0])
    other_br
    s2 ='; 累计回报：{}，累计标差：{}'.format(br.sum(), br.std())
    print(s,s2)
    pass

if __name__ == '__main__':
    '''
    sim
    数据中心
    经纪人
    '''

    evaluateSchedule(MinBalanceSchedule())
    evaluateSchedule(DRF())
    evaluateSchedule(RandomMcSchedule())
    evaluateSchedule(DoubleRandomSchedule())
    t1 = Schedule()
    t1.training = False
    evaluateSchedule(t1)

    mc_schedule = Schedule()
    memory = SimpleA2CMemory()
    print('******************')
    mc_schedule.training = False
    evaluateSchedule(mc_schedule)
    mc_schedule.training = True


    for n_epi in range(DRLParameters.epochs):
        env = SimEnv()
        b = Base(env)
        dc = DataCenter(env, Parameters.DC_config, mc_schedule=mc_schedule)
        broker = Broker(env)
        env.run()

        over_tasks = []
        for dc_id in broker.dc_id_set:
            t_dc = env.obj_list[dc_id]
            for t_clu in t_dc.cluster_list:
                # Reward.reviseTrack(t_clu)
                memory.handle_track(t_clu.drl_track)
                for t_mc in t_clu.machine_list:
                    over_tasks += t_mc.over_task_list
        # Score.taskSLAScore(over_tasks)
        mc_schedule.train(memory)

        memory.clear()
        if n_epi%DRLParameters.train_step == 0:
            # if memory.buffSize() > DRLParameters.buffer_mini_size:
            #     # for i in range(10):
            #     #     mc_schedule.train(ddqn_memory)
            #
            #     # ddqn_memory.clear()
            mc_schedule.saveModel()
            mc_schedule.training = False
            evaluateSchedule(mc_schedule)
            mc_schedule.training = True
        # if n_epi%DRLParameters.updata_target_step == 0:
        #     mc_schedule.updataTarget()
    pass