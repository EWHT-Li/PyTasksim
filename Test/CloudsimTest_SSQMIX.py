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
from SpaceShareSchedul.QMIX import QMIXMemory, Schedule
from Schedul import Reward

from SpaceShareSchedul.Min import MinMakespanSchedule
from SpaceShareSchedul.MinMin import MinMinMakespanSchedule
from SpaceShareSchedul.Random import RandomSchedule
from SpaceShareSchedul.RoundRobin import RoundRobinSchedule
from SpaceShareSchedul.PSO import PSOSchedule

import pandas as pd

def evaluateSchedule(schdule):
    env = SimEnv()
    b = Base(env)
    dc = DataCenter(env, Parameters.DC_config, mc_schedule=schdule)
    broker = Broker(env)
    env.run()
    env.clock
    over_tasks = []
    r1_score = 0
    for dc_id in broker.dc_id_set:
        t_dc = env.obj_list[dc_id]
        for t_clu in t_dc.cluster_list:
            t_r1 = Reward.reviseTrackWithExpendAndMakespan(t_clu)
            r1_score += t_r1
            for t_mc in t_clu.machine_list:
                over_tasks += t_mc.over_task_list
    abs_sla, comp_sla = Score.taskSLAScore(over_tasks)
    bl_score = Score.dcBalanceScore(dc)
    lb_score = Score.dcLoadBalnace(dc)
    expend_score = Score.dcExpendScore(dc)
    task_time_score = Score.dcTaskTimeScore(dc)
    total_r, total_gae_r, l = Score.dcRewardTotal_TotalGae_Tasklen(dc)
    response_time = Score.taskResponseScore(over_tasks)

    s = '{}:花费：{}；完成时间：{}；任务绝对SLA得分：均{}，标{}；任务相对SLA得分：均{}，标{}；均衡得分：{}'.format(schdule.__class__.__name__, expend_score,
                                                                        env.clock, abs_sla.mean(), abs_sla.std(),
                                                                        comp_sla.mean(), comp_sla.std(), bl_score)

    s2 ='; 累计回报：{}，总累计回报：{}，任务数：{}'.format(total_r, total_gae_r, l)
    # print(s,s2)

    result = {
        'total_r': total_r,
        'total_gae_r': total_gae_r,
        'task_num': l,
        'loadBalance': bl_score,
        'end_time': env.clock,
        'expend_score': expend_score,
        'sla_mean': abs_sla.mean(),
        'sla_std':abs_sla.std(),
        'response_time_mean': response_time.mean(),
        'response_time_srd': response_time.std(),
        'response_time_std': response_time.std(),
        'task_time_score': task_time_score,
        'r1_score': r1_score,
        'lb_score': lb_score,
    }
    print(schdule.__class__.__name__, result)
    return result

def loadModelevaluateSchedule():
    mc_schedule = Schedule()
    mc_schedule.loadModel('DDQNOne/2021-03-07-13-54-DDQNOneE1.pkl')
    mc_schedule.training = False
    evaluateSchedule(mc_schedule)
    mc_schedule.training = True

import os, time
import numpy as np
import sys

if __name__ == '__main__':
    Parameters.r1_rate = float(sys.argv[1])
    Parameters.r2_rate = float(sys.argv[2])
    print(Parameters.r1_rate, Parameters.r2_rate)
    PSO_result = evaluateSchedule(PSOSchedule())
    min_makespan_result = evaluateSchedule(MinMakespanSchedule())
    minmin_makespan_result = evaluateSchedule(MinMinMakespanSchedule())
    rand_result = evaluateSchedule(RandomSchedule())
    RR_result = evaluateSchedule(RoundRobinSchedule())
    result1 = [min_makespan_result, minmin_makespan_result, rand_result, RR_result, PSO_result]
    columns1 = Parameters.SCOREResult
    index1 = ['min_makespan', 'minmin_makespan', 'Random', 'RR', 'PSO']
    df1 = pd.DataFrame(result1, columns=columns1, index=index1)

    # raise Exception('GG')
    '''
    sim
    数据中心
    经纪人
    '''
    Parameters.ShareModel = Parameters.VMState.SPACESHARE

    mc_schedule = Schedule()
    ddqn_memory = QMIXMemory()

    path = Parameters.SavePath + Schedule.ModelSaveName + '/' + mc_schedule.getTimeStr() +'/'
    if not os.path.exists(path):
        os.makedirs(path)
    print(mc_schedule.getTimeStr())
    Universal.saveClass(Parameters, path, filt=['ShareModel'])
    Universal.saveClass(DRLParameters, path)
    Universal.saveClass(TaskGenerateParam, path)
    df1.to_csv(path_or_buf=path + '/basic.csv')

    # mc_schedule.loadModel(path='D:/Code/li-Cloudsim/ModelSave/SDDQN/2021-03-27-22-43d/2021-03-27-22-43d-DDQNOne.pkl') #2021-03-27-22-43d
    # mc_schedule.updataTarget()

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
    loss = []
    task_t = []


    train_time = []
    for n_epi in range(DRLParameters.epochs):
        env = SimEnv()
        dc = DataCenter(env, Parameters.DC_config, mc_schedule=mc_schedule)
        broker = Broker(env)
        env.run()


        over_tasks = []
        for dc_id in broker.dc_id_set:
            t_dc = env.obj_list[dc_id]
            for t_clu in t_dc.cluster_list:
                Reward.reviseTrackWithExpendAndMakespan(t_clu)
                ddqn_memory.handle_track(t_clu.drl_track)
                for t_mc in t_clu.machine_list:
                    over_tasks += t_mc.over_task_list
        # Score.taskSLAScore(over_tasks)


        # if n_epi%DRLParameters.train_step == 0: and n_epi%DRLParameters.train_step == 0
        if ddqn_memory.buffSize() > DRLParameters.buffer_mini_size :
            # for i in range(10):
            #     mc_schedule.train(ddqn_memory)
            now1 = time.time()
            t_loss = mc_schedule.train(ddqn_memory)
            now2 = (time.time())
            train_time.append(now2-now1)
            print('train_one_time:', now2-now1)
            # ddqn_memory.clear()

            mc_schedule.training = False
            now1 = (time.time())
            eva_result = evaluateSchedule(mc_schedule)
            eva_result['loss'] = t_loss
            now2 = (time.time())
            print('eva_one_time:', now2 - now1)
            result.append(eva_result)

            t_r.append(eva_result['total_r'])
            t_g_r.append(eva_result['response_time_mean'])
            e_t.append(eva_result['end_time'])
            expend.append(eva_result['expend_score'])
            loss.append(eva_result['loss'])
            task_t.append(eva_result['r1_score'])
            ax[0][0].cla()
            ax[0][1].cla()
            ax[0][2].cla()
            ax[1][0].cla()
            ax[1][1].cla()
            ax[1][2].cla()
            ax[0][0].plot(t_r, label='ToTal_Reward')
            ax[0][1].plot(t_g_r, label='ToTal_Reward2')
            ax[0][2].plot(e_t, label='ToTal_Reward3')
            ax[1][0].plot(expend, label='ToTal_Reward3')
            ax[1][1].plot(loss, label='ToTal_Reward4')
            ax[1][2].plot(task_t, label='ToTal_Reward5')
            # plt.show()
            plt.pause(0.0000001)
            mc_schedule.training = True
        if n_epi%DRLParameters.updata_target_step == 0:
            mc_schedule.updataTarget()
        if n_epi%10 == 0 and len(result) > 0:
            df = pd.DataFrame(columns=Parameters.SCOREResult)
            df = df.append(result, ignore_index=True)
            df.to_csv(path_or_buf=path+'/data.csv' , index=None)
            mc_schedule.saveModel()
            print('#######################{}##############'.format(np.array(train_time).mean()))
            pass
    # plt.show()
    pass