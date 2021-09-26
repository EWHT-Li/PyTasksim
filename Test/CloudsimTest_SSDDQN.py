from Base.SimEnv import SimEnv
from Base.Parameters import Parameters, DRLParameters, TaskGenerateParam
from Modules.DataCenter import DataCenter
from Modules.Broker import Broker
from Base.SimBaseInterfaces import Base
from Modules.Cluster import Cluster
from Tool import Score
from Tool.Universal import Universal
import matplotlib.pyplot as plt
'''
DDQN
'''
from SpaceShareSchedul.SSDDQN import DDQNOneMemory, Schedule
from Schedul import Reward

from SpaceShareSchedul.Min import MinMakespanSchedule
from SpaceShareSchedul.MinMin import MinMinMakespanSchedule
from SpaceShareSchedul.Random import RandomSchedule
from SpaceShareSchedul.RoundRobin import RoundRobinSchedule

import pandas as pd

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
            Reward.reviseTrackWithExpendAndMakespan(t_clu)
            for t_mc in t_clu.machine_list:
                over_tasks += t_mc.over_task_list
    abs_sla, comp_sla = Score.taskSLAScore(over_tasks)
    bl_score = Score.dcBalanceScore(dc)
    expend_score = Score.dcExpendScore(dc)
    # br,br1,br2,ext = Score.cluRewardScore(dc.cluster_list[0])
    # other_br = Score.cluOtherRewardScore(dc.cluster_list[0])
    total_r, total_gae_r, l = Score.dcRewardTotal_TotalGae_Tasklen(dc)

    s = '{}:花费：{}；完成时间：{}；任务绝对SLA得分：均{}，标{}；任务相对SLA得分：均{}，标{}；均衡得分：{}'.format(schdule.__class__.__name__, expend_score,
                                                                        env.clock, abs_sla.mean(), abs_sla.std(),
                                                                        comp_sla.mean(), comp_sla.std(), bl_score)

    s2 ='; 累计回报：{}，总累计回报：{}，任务数：{}'.format(total_r, total_gae_r, l)
    print(s,s2)

    result = {
        'total_r': total_r,
        'total_gae_r': total_gae_r,
        'task_num': l,
        'loadBalance': bl_score,
        'end_time': env.clock,
        'expend_score': expend_score,
        'sla_mean': abs_sla.mean(),
        'sla_std':abs_sla.std(),
    }
    return total_r, total_gae_r, l, bl_score, env.clock, expend_score

def loadModelevaluateSchedule():
    mc_schedule = Schedule()
    mc_schedule.loadModel('DDQNOne/2021-03-07-13-54-DDQNOneE1.pkl')
    mc_schedule.training = False
    evaluateSchedule(mc_schedule)
    mc_schedule.training = True

import os

if __name__ == '__main__':
    # evaluateSchedule(MinMakespanSchedule())
    # # loadModelevaluateSchedule()
    # evaluateSchedule(MinMinMakespanSchedule())
    # evaluateSchedule(RandomSchedule())
    # evaluateSchedule(RoundRobinSchedule())

    raise Exception('GG')
    '''
    sim
    数据中心
    经纪人
    '''
    Parameters.ShareModel = Parameters.VMState.SPACESHARE

    # evaluateSchedule(MinMakespanSchedule())
    # evaluateSchedule(DRF())
    # evaluateSchedule(RandomMcSchedule())
    # evaluateSchedule(DoubleRandomSchedule())
    t1 = Schedule()
    t1.training = False
    evaluateSchedule(t1)

    path = Parameters.SavePath + Schedule.ModelSaveName + '/' + t1.getTimeStr() +'/'
    if not os.path.exists(path):
        os.makedirs(path)
    Universal.saveClass(Parameters, path)
    Universal.saveClass(DRLParameters, path)
    Universal.saveClass(TaskGenerateParam, path)

    mc_schedule = Schedule()
    ddqn_memory = DDQNOneMemory()
    print('******************')
    mc_schedule.training = False
    evaluateSchedule(mc_schedule)
    mc_schedule.training = True

    fig, ax = plt.subplots(2,3)
    t_r = []
    t_g_r = []
    e_t = []
    expend = []
    result = []



    for n_epi in range(DRLParameters.epochs):
        env = SimEnv()


        dc = DataCenter(env, Parameters.DC_config, mc_schedule=mc_schedule)
        broker = Broker(env)
        env.run()
        # total_r, total_gae_r, l = Score.dcRewardTotal_TotalGae_Tasklen(dc)
        # ax[0].cla()
        # ax[1].cla()
        # ax[0].plot(t_r, label='ToTal_Reward')
        # ax[1].plot(t_g_r, label='ToTal_Reward')
        # plt.pause(0.0000001)
        # t_r.append(total_r)
        # t_g_r.append(total_gae_r)

        over_tasks = []
        for dc_id in broker.dc_id_set:
            t_dc = env.obj_list[dc_id]
            for t_clu in t_dc.cluster_list:
                Reward.reviseTrackWithExpendAndMakespan(t_clu)
                ddqn_memory.handle_track(t_clu.drl_track)
                for t_mc in t_clu.machine_list:
                    over_tasks += t_mc.over_task_list
        # Score.taskSLAScore(over_tasks)


        # if n_epi%DRLParameters.train_step == 0:
        if ddqn_memory.buffSize() > DRLParameters.buffer_mini_size:
            # for i in range(10):
            #     mc_schedule.train(ddqn_memory)
            mc_schedule.train(ddqn_memory)
            # ddqn_memory.clear()

            mc_schedule.training = False
            eva_result = evaluateSchedule(mc_schedule)
            result.append(eva_result)
            total_r, total_gae_r, l, bl_score, et, expend_score = list(eva_result.values())
            print(ddqn_memory.buffSize())
            t_r.append(total_r)
            t_g_r.append(total_gae_r)
            e_t.append(et)
            expend.append(expend_score)
            ax[0][0].cla()
            ax[0][1].cla()
            ax[0][2].cla()
            ax[1][0].cla()
            ax[0][0].plot(t_r, label='ToTal_Reward')
            ax[0][1].plot(t_g_r, label='ToTal_Reward2')
            ax[0][2].plot(e_t, label='ToTal_Reward3')
            ax[1][0].plot(expend, label='ToTal_Reward3')
            # plt.show()
            plt.pause(0.0000001)
            mc_schedule.training = True
        if n_epi%DRLParameters.updata_target_step == 0:
            mc_schedule.updataTarget()
        if n_epi%100 == 0:
            df = pd.DataFrame(columns=Parameters.SCOREResult)
            df.append(result, ignore_index=True)
            df.to_csv(path_or_buf=path+'/data.csv' , index=None)
            mc_schedule.saveModel()
            pass
    plt.show()
    pass