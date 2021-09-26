from Base.SimEnv import SimEnv
from Base.Parameters import Parameters, DRLParameters, TaskGenerateParam
from Modules.DataCenter import DataCenter
from Modules.Broker import Broker
from Base.SimBaseInterfaces import Base
from Tool import Score
from Schedul import Reward



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
    return result