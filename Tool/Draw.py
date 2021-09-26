# 加载数据  绘图   加载参数

import pandas as pd

from Base.SimEnv import SimEnv
from Modules.DataCenter import DataCenter
from Modules.Broker import Broker
from Base.SimBaseInterfaces import Base
from Tool import Score
from Tool.Universal import Universal
import matplotlib.pyplot as plt
from Schedul import Reward
import numpy as np
import seaborn as sns


from SpaceShareSchedul.Min import MinMakespanSchedule
from SpaceShareSchedul.MinMin import MinMinMakespanSchedule
from SpaceShareSchedul.Random import RandomSchedule
from SpaceShareSchedul.RoundRobin import RoundRobinSchedule
from SpaceShareSchedul.PSO import PSOSchedule

def drawCompare(path,score_name = ['total_r', 'end_time', 'expend_score', 'response_time_mean'],score_title = ['Reward', 'Makespan', 'Expend Score', 'Response Time']):
    data_path = '{}/{}'.format(path, 'data.csv')
    basic_path = '{}/{}'.format(path, 'basic.csv')

    data_f = open(data_path, mode='r')
    df_data = pd.read_csv(data_f)
    data_f.close()

    basic_f = open(basic_path, mode='r')
    df_basic = pd.read_csv(basic_f, index_col=[0])
    basic_f.close()

    '''
    要画什么图呢？
    1.回报曲线对比
    '''
    r = df_data.columns[0]
    total_r = df_data[r].values


    df_data['expend_score'] = df_data['expend_score']/3600
    df_basic['expend_score'] = df_basic['expend_score']/3600

    score_name_t = {}
    for s in score_name:
        score_name_t[s+'_t'] = []
    num = []
    for i in range(1, len(df_data['total_r'].values)-2):
        for j in range(3):
            num.append(i)
            for s in score_name:
                score_name_t[s+'_t'].append(df_data[s].values[i+j])
    df_data['num'] = pd.Series(num)
    for s in score_name:
        df_data[s + '_t'] = pd.Series(score_name_t[s+'_t'])
    score_name_t['num'] = num
    df_data2 = pd.DataFrame(score_name_t)


    fig, ax = plt.subplots(2, 2)

    ax_list = ax.reshape(-1)

    # sns.lineplot(x="num", y='total_r_t', label="DDQN", data=df_data, ax=ax_list[0])
    for ax_score in zip(ax_list, score_name):
        x = np.array(list(range(0, len(df_data[ax_score[1]].values))))
        y = df_data[ax_score[1]].values
        ax_score[0].plot(x, y, label="DDQN", color="lightskyblue")
        # parameter = np.polyfit(x, y, 28)
        # p = np.poly1d(parameter)
        # ax_score[0].plot(x, p(x), color='fuchsia',label="DDQN-s")

        # sns.lineplot(x="num", y=ax_score[1] + '_t', label="DDQN", data=df_data2, ax=ax_score[0])

    '''
    ax[0][0].plot(total_r, label='DDQN')
    ax[0][1].plot(df_data['end_time'].values, label="DDQN")
    ax[1][0].plot(df_data['expend_score'].values, label='DDQN')
    ax[1][1].plot(df_data['sla_mean'].values, label="DDQN")
    '''

    for ax_score in zip(ax_list, score_name):
        for sn_v in zip(df_basic.index, df_basic[ax_score[1]].values):
            if sn_v[0] != 'min_makespan':
                ax_score[0].plot([sn_v[1]] * len(total_r), label=sn_v[0])
    '''
        for data in zip(df_basic.index, df_basic[r].values,
                    df_basic['end_time'].values, df_basic['expend_score'].values,
                    df_basic['sla_mean'].values,
                    ):
        label = data[0]
        v_r = data[1]
        ax[0][0].plot([v_r] * len(total_r), label=label)
        ax[0][1].plot([data[2]] * len(total_r), label=label)
        ax[1][0].plot([data[3]] * len(total_r), label=label)
        ax[1][1].plot([data[4]] * len(total_r), label=label)
    '''

    #设置图例
    for ax_title in zip(ax_list, score_title):
        ax_title[0].legend(bbox_to_anchor=(0, 0.65, .46, 0.3),
                           ncol=1, mode="expand", borderaxespad=0.)
        ax_title[0].set_title(ax_title[1])
        ax_title[0].set_xlabel('epoch')

    '''
    # ax[0][0].legend(bbox_to_anchor=(0, 0.65, .46, 0.3),
    #        ncol=1, mode="expand", borderaxespad=0.) #loc=3
    # ax[0][0].set_title('Reward')
    # ax[0][0].set_xlabel('epoch')
    #
    # ax[0][1].legend(bbox_to_anchor=(0, 0.65, .46, 0.3),
    #                 ncol=1, mode="expand", borderaxespad=0.)
    # ax[0][1].set_title('Makespan')
    # ax[0][1].set_xlabel('epoch')
    #
    # ax[1][0].legend(bbox_to_anchor=(0, 0.65, .46, 0.3),
    #                 ncol=1, mode="expand", borderaxespad=0.)
    # ax[1][0].set_title('Expend Score')
    # ax[1][0].set_xlabel('epoch')
    #
    # ax[1][1].legend(bbox_to_anchor=(0, 0.65, .46, 0.3),
    #                 ncol=1, mode="expand", borderaxespad=0.)
    # ax[1][1].set_title('Response Time')
    # ax[1][1].set_xlabel('epoch')
'''

    plt.show()

def drawCompares(paths: list, basic_path = None, size = 400, names=None,
                 score_name = ['total_r', 'end_time', 'expend_score', 'response_time_mean'],
                 score_title = ['Reward', 'Makespan', 'Expend Score', 'Response Time'],
                 dqn_name= ['1-9','5-5','9-1']):
    if names == None:
        names = paths

    if basic_path == None:
        basic_path = paths[0]
    basic_path = '{}/{}'.format(basic_path, 'basic.csv')
    basic_f = open(basic_path, mode='r')
    df_basic = pd.read_csv(basic_f, index_col=[0])
    basic_f.close()

    #为了加入pso元素
    pso_result = evaluateSchedule(PSOSchedule())
    pso_s = pd.Series(pso_result)
    index_list = df_basic.index.to_list()
    df_basic = df_basic.append(pso_s, ignore_index=True)
    index_list.append('PSO')
    df_basic.insert(0, 'index', index_list)
    df_basic.set_index('index', drop=True, append=False, inplace=True)





    df_basic['expend_score'] = df_basic['expend_score'] / 3600

    df_data_s = []
    for path in paths:
        data_path = '{}/{}'.format(path, 'data.csv')
        data_f = open(data_path, mode='r')
        df_data = pd.read_csv(data_f)
        data_f.close()
        df_data_s.append(df_data)
        df_data['expend_score'] = df_data['expend_score'] / 3600

    '''
    要画什么图呢？
    1.回报曲线对比
    '''


    score_name_t = {}
    for s in score_name:
        score_name_t[s+'_t'] = []
    num = []
    for i in range(1, size-2):
        for j in range(3):
            num.append(i)
            for s in score_name:
                score_name_t[s+'_t'].append(df_data[s].values[i+j])
    df_data['num'] = pd.Series(num)
    for s in score_name:
        df_data[s + '_t'] = pd.Series(score_name_t[s+'_t'])
    score_name_t['num'] = num
    df_data2 = pd.DataFrame(score_name_t)

    fig, ax = plt.subplots(2, 2)
    ax_list = ax.reshape(-1)

    for name, df_data, dname in zip(names, df_data_s, dqn_name):
        for ax_score in zip(ax_list, score_name):
            x = np.array(list(range(0, size)))
            y = df_data[ax_score[1]].values[:size]
            ax_score[0].plot(x, y, label="DDQN"+dname) #, color="lightskyblue"
        pass

    # df_data = df_data_s[0]
    # sns.lineplot(x="num", y='total_r_t', label="DDQN", data=df_data, ax=ax_list[0])
    # for ax_score in zip(ax_list, score_name):
    #     x = np.array(list(range(0, size)))
    #     y = df_data[ax_score[1]].values[:size]
    #     ax_score[0].plot(x, y, label="DDQN", ) # color="lightskyblue"
    #     parameter = np.polyfit(x, y, 28)
    #     p = np.poly1d(parameter)
    #     ax_score[0].plot(x, p(x), color='fuchsia',label="DDQN-s")

        # sns.lineplot(x="num", y=ax_score[1] + '_t', label="DDQN", data=df_data2, ax=ax_score[0])


    for ax_score in zip(ax_list, score_name):
        for sn_v in zip(df_basic.index, df_basic[ax_score[1]].values):
            if sn_v[0] != 'min_makespan':
                if sn_v[0] == 'minmin_makespan':
                    ax_score[0].plot([sn_v[1]] * size, label='MinMin')
                else:
                    ax_score[0].plot([sn_v[1]] * size, label=sn_v[0])

    #设置图例
    i = 0
    for ax_title in zip(ax_list, score_title):
        i += 1
        if i == 4:
            ax_title[0].legend(bbox_to_anchor=(0.5, 0.55, 0.46, 0.3),
                           ncol=1, mode="expand", borderaxespad=0.)
        ax_title[0].set_title('({0}) '.format(chr(96+i))+ax_title[1])
        ax_title[0].set_xlabel('epoch')
        ax_title[0].set_ylabel(ax_title[1])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

def reloadParame(path):
    parame_file_path = '{}/{}.json'.format(path, 'Parameters')
    drlparame_file_path = '{}/{}.json'.format(path, 'DRLParameters')
    taskgenparame_file_path = '{}/{}.json'.format(path, 'TaskGenerateParam')
    print(Parameters.MipsScale)
    parame = Universal.loadClass(Parameters, parame_file_path)
    drl_parame = Universal.loadClass(DRLParameters, drlparame_file_path)
    task_parame = Universal.loadClass(TaskGenerateParam, taskgenparame_file_path)

    # 输出  VM数量  任务长度   回报比例
    result ={
        'vm_count': len(Parameters.MS1),
        'Task_epoch': len(TaskGenerateParam.E),
        'reward_rate':{'makespan_r1':Parameters.r1_rate, 'expend_r2':Parameters.r2_rate}
    }
    print(result)

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
        # 'lb_score': lb_score,
    }
    print(schdule.__class__.__name__, result)
    return result

def evaluateSchedule_old(schdule):
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
    response_time = Score.taskResponseScore(over_tasks)
    bl_score = Score.dcBalanceScore(dc)
    expend_score = Score.dcExpendScore(dc)
    total_r, total_gae_r, l = Score.dcRewardTotal_TotalGae_Tasklen(dc)

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
    }
    print(schdule.__class__.__name__, result)
    return result

from SpaceShareSchedul.SSDDQNGPU import DDQNOneMemory, Schedule
def loadModelevaluateSchedule(path):
    mc_schedule = Schedule()
    mc_schedule.loadModel(path=path)
    mc_schedule.training = False
    n1 = int(time.time()*1000)
    result = evaluateSchedule(mc_schedule)
    n2 = int(time.time()*1000)
    mc_schedule.training = True
    print("DRL-time:", n2 - n1)
    return result

def drawCompareWithOtherModel(path, model_path):
    # 加载path 的模型效果和 参数 加载p2的模型
    reloadParame(path)
    load_model_result = loadModelevaluateSchedule(model_path)


    data_path = '{}/{}'.format(path, 'data.csv')
    basic_path = '{}/{}'.format(path, 'basic.csv')

    data_f = open(data_path, mode='r')
    df_data = pd.read_csv(data_f)
    data_f.close()

    basic_f = open(basic_path, mode='r')
    df_basic = pd.read_csv(basic_f, index_col=[0])
    basic_f.close()

    '''
    要画什么图呢？
    1.回报曲线对比
    '''
    r = df_data.columns[0]
    total_r = df_data[r].values
    score_name = ['total_r', 'end_time', 'expend_score', 'response_time_mean']
    score_title = ['Reward', 'Makespan', 'Expend Score', 'Response Time']

    df_data['expend_score'] = df_data['expend_score']/3600
    df_basic['expend_score'] = df_basic['expend_score']/3600
    fig, ax = plt.subplots(2, 2)

    ax_list = ax.reshape(-1)

    for ax_score in zip(ax_list, score_name):
        x = np.array(list(range(1, len(df_data[ax_score[1]].values)+1)))
        y = df_data[ax_score[1]].values
        ax_score[0].plot(x, y, label="DDQN", color="lightskyblue")
        # ax_score[0].plot(df_data[ax_score[1]].values, label="DDQN")

        parameter = np.polyfit(x, y, 25)
        p = np.poly1d(parameter)
        ax_score[0].plot(x, p(x), color='fuchsia',label="DDQN-s")
        # plt.show()
        print('??:',load_model_result[ax_score[1]])
        ax_score[0].plot(x, [load_model_result[ax_score[1]]]*len(x), label="other-DDQN")


    for ax_score in zip(ax_list, score_name):
        for sn_v in zip(df_basic.index, df_basic[ax_score[1]].values):
            ax_score[0].plot([sn_v[1]] * len(total_r), label=sn_v[0])

    #设置图例
    for ax_title in zip(ax_list, score_title):
        ax_title[0].legend(bbox_to_anchor=(0, 0.65, .46, 0.3),
                           ncol=1, mode="expand", borderaxespad=0.)
        ax_title[0].set_title(ax_title[1])
        ax_title[0].set_xlabel('epoch')

    plt.show()

# def drawModelWithTheEnv(env_path, model_path):
#     reloadParame(path)
#     load_model_result = loadModelevaluateSchedule(model_path)
#
#     data_path = '{}/{}'.format(path, 'data.csv')
#     basic_path = '{}/{}'.format(path, 'basic.csv')
#
#     data_f = open(data_path, mode='r')
#     df_data = pd.read_csv(data_f)
#     data_f.close()
#
#     basic_f = open(basic_path, mode='r')
#     df_basic = pd.read_csv(basic_f, index_col=[0])
#     basic_f.close()
#
#     '''
#     要画什么图呢？
#     1.回报曲线对比
#     '''
#     r = df_data.columns[0]
#     total_r = df_data[r].values
#     score_name = ['total_r', 'end_time', 'expend_score', 'sla_mean']
#     score_title = ['Reward', 'Makespan', 'Expend Score', 'Response Time']
#
#     df_data['expend_score'] = df_data['expend_score'] / 3600
#     df_basic['expend_score'] = df_basic['expend_score'] / 3600
#     fig, ax = plt.subplots(2, 2)
#
#     ax_list = ax.reshape(-1)
#
#     for ax_score in zip(ax_list, score_name):
#         x = np.array(list(range(1, len(df_data[ax_score[1]].values) + 1)))
#         y = df_data[ax_score[1]].values
#         ax_score[0].plot(x, y, label="DDQN", color="lightskyblue")
#         # ax_score[0].plot(df_data[ax_score[1]].values, label="DDQN")
#
#         parameter = np.polyfit(x, y, 25)
#         p = np.poly1d(parameter)
#         ax_score[0].plot(x, p(x), color='fuchsia', label="DDQN-s")
#         # plt.show()
#         print('??:', load_model_result[ax_score[1]])
#         ax_score[0].plot(x, [load_model_result[ax_score[1]]] * len(x), label="other-DDQN")
#
#     for ax_score in zip(ax_list, score_name):
#         for sn_v in zip(df_basic.index, df_basic[ax_score[1]].values):
#             ax_score[0].plot([sn_v[1]] * len(total_r), label=sn_v[0])
#
#     # 设置图例
#     for ax_title in zip(ax_list, score_title):
#         ax_title[0].legend(bbox_to_anchor=(0, 0.65, .46, 0.3),
#                            ncol=1, mode="expand", borderaxespad=0.)
#         ax_title[0].set_title(ax_title[1])
#         ax_title[0].set_xlabel('epoch')
#
#     plt.show()
import time,random
def testEvaSchedule():
    n1 = int(time.time()*1000)
    min_makespan_result = evaluateSchedule(MinMakespanSchedule())
    n2 = int(time.time()*1000)
    print("min-time:", n2 - n1)
    minmin_makespan_result = evaluateSchedule(MinMinMakespanSchedule())
    rand_result = evaluateSchedule(RandomSchedule())
    n1 = int(time.time() * 1000)
    RR_result = evaluateSchedule(RoundRobinSchedule())
    n2 = int(time.time() * 1000)
    print("RR-time:", n2 - n1)
    n1 = int(time.time()*1000)
    pso_result = evaluateSchedule(PSOSchedule())
    n2 = int(time.time()*1000)
    print("PSO-time:", n2 - n1)


from Base.Parameters import Parameters, DRLParameters, TaskGenerateParam
import inspect
import re

if __name__ == '__main__':
    # fmri = sns.load_dataset("fmri")
    d1 = '2021-04-09-09-19' #'2021-03-27-21-57g' h:2021-04-09-10-09  m:2021-04-09-09-19
    # path = '{}/SDDQN/{}'.format(Parameters.SavePath,d1)
    #
    # drawCompare(path,
    #             score_name=['expend_score', 'response_time_mean', 'task_time_score', 'r1_score'],
    #             score_title=['Expend Score', 'Response Time', 'Task Time', 'R1']
    #             )
    # print('gg')

    # reloadParame(path)
    # '2021-03-31-20-19'0.7 x  '2021-03-31-20-39'0.1xd
    # names = ['2021-03-31-20-18', '2021-03-31-20-40', '2021-03-31-21-18', '2021-03-31-20-20', '2021-03-31-20-57',]

    # names = ['2021-03-31-20-18', '2021-04-01-14-34', '2021-04-01-15-10', ]

    # names = ['2021-04-01-15-52','2021-04-01-14-34', '2021-04-01-15-10', ]
    # names = ['2021-04-06-23-21', '2021-04-06-23-09', '2021-04-06-23-10']
    # names = ['2021-04-06-23-30','2021-04-07-11-07' ,'2021-04-06-23-44','2021-04-07-11-09', '2021-04-06-23-32']
    # names = ['2021-04-07-21-56', '2021-04-07-16-41', '2021-04-07-22-26', '2021-04-08-02-10', '2021-04-08-02-10']
    names = ['2021-04-07-21-56', '2021-04-07-22-26', '2021-04-08-09-18'] #result **************
    # names = ['2021-04-10-16-46', '2021-04-10-16-47', '2021-04-10-16-48']
    # names = [d1,]
    # names = ['2021-05-08-15-06-00', '2021-05-08-15-06-41', '2021-05-08-15-07-11']
    paths = ['{}/SDDQN/{}'.format(Parameters.SavePath,d1) for d1 in names]

    for path in paths:
        reloadParame(path)
        names.append('{}'.format(Parameters.r1_rate))

    loadModelevaluateSchedule('E:/Storage/Code/Code/li-Cloudsim/ModelSave/SDDQN/2021-04-09-09-19/2021-04-09-09-19-DDQNOne.pkl')
    testEvaSchedule()

    drawCompares(paths, names=names,size=200,
                 score_name = ['expend_score', 'response_time_mean', 'r1_score', 'end_time'], #'task_time_score', 'total_r',
                 score_title=['Expend Score', 'Overdue Time', 'Throughput Rate', 'Makespan'],
                 dqn_name=['', '1-9','5-5','9-1']) #'Task Time', 'Reward',

    # d2 = '2021-03-27-22-43d'
    # drawCompareWithOtherModel(path, 'D:/Code/li-Cloudsim/ModelSave/SDDQN/{}/{}-DDQNOne.pkl'.format(d2,d2))


    # E = []
    # for i in range(0,130):
    #     E.append(random.randint(1,1000))
    # reloadParame(path)
    # TaskGenerateParam.E = E
    # TaskGenerateParam.withE()
    # testEvaSchedule()
    # TaskGenerateParam.index = 0
    # loadModelevaluateSchedule('D:/Code/li-Cloudsim/ModelSave/SDDQN/{}/{}-DDQNOne.pkl'.format(d1,d1))

    '''
    参数载入设置
    '''
    # testEvaSchedule()
    # print("#####################")

    # testEvaSchedule()
    # 拟合曲线， 非同环境测试
    pass