import numpy as np
from Base.Parameters import DRLParameters

def taskSLAScore(tasks):
    abs_sla = []
    comp_sla = []
    for task in tasks:
        abs_sla.append(task.start_time - task.load_time)
        comp_sla.append((task.start_time - task.load_time)/task.duration)
    return np.array(abs_sla), np.array(comp_sla)

def taskResponseScore(tasks):
    abs_sla = []
    comp_sla = []
    for task in tasks:
        abs_sla.append(max(task.finish_time - task.load_time - task.duration, 0))
    result = np.array(abs_sla)
    return result

def cluBalanceScore(cluster):
    recode_list =cluster.recode_list
    if len(recode_list) < 2:
        raise Exception('记录至少有2条的哇')
    l = len(recode_list)
    i = 1
    score = 0
    while i < l:
        score += (recode_list[i]['time'] - recode_list[i-1]['time'])*recode_list[i-1]['balance']
        i += 1
    return score

def cluRewardScore(cluster):
    reward_list = cluster.reward_list
    score = 0
    br = []
    br1 = []
    br2 = []
    ext = []
    for track in cluster.drl_track:
        br.append(track['r'])
    for reward in reward_list:
        r1,r2,r3,r4 = reward
        # br.append(r1)
        br1.append(r2)
        br2.append(r3)
        ext.append(r4)
        pass
    return np.array(br), np.array(br1), np.array(br2), np.array(ext)

def cluOtherRewardScore(cluster):
    other_choose = cluster.other_choose
    return other_choose

def dcBalanceScore(dc):
    cluster_list = dc.cluster_list
    score = 0
    for clu in cluster_list:
        score += cluBalanceScore(clu)
    return score

def dcRewardTotal_TotalGae_Tasklen(dc):
    cluster_list = dc.cluster_list
    total_r, total_gae_r, l = 0, 0, 0
    for clu in cluster_list:
        t_total_r, t_total_gae_r, t_l =handle_track(clu.drl_track)
        total_r += t_total_r
        total_gae_r += t_total_gae_r
        l += t_l
    return total_r, total_gae_r, l


def cluExpendScore(clu):
    result = 0
    for t_mc in clu.machine_list:
        for t_task in t_mc.over_task_list:
            result += t_task.expend
    return result
def dcExpendScore(dc):
    cluster_list = dc.cluster_list
    score = 0
    for clu in cluster_list:
        score += cluExpendScore(clu)
    return score

def cluTaskTimeScore(clu):
    result = 0
    for t_mc in clu.machine_list:
        for t_task in t_mc.over_task_list:
            result += t_task.finish_time - t_task.start_time
    return result

def dcTaskTimeScore(dc):
    cluster_list = dc.cluster_list
    score = 0
    for clu in cluster_list:
        score += cluTaskTimeScore(clu)
    return score

def cluR1Score(cluster):
    tracks = cluster.drl_track
    # recodes = cluster.recode_list
    result = 0
    for item in tracks:
        if item['task_recode']['r'] == None:
            start_s = []
            finish_s = []
            mipsl_s = []
            bwl_s = []
            # slat_s = []
            # expend_s = []
            # slarate_s = []
            # sla_abs_s = []
            for task in item['task_recode']['task_list']:
                start_s.append(task.start_time)
                finish_s.append(task.finish_time)
                mipsl_s.append(task.mips_length)
                bwl_s.append(task.bw_length)
                # sla_abs_s.append(task.finish_time - task.load_time - task.duration)
                # slat_s.append(max(task.finish_time - task.load_time - task.duration, 0))
                # slarate_s.append((task.start_time-task.load_time)/(task.finish_time-task.load_time))
                # expend_s.append(task.expend)

            r1 = (sum(mipsl_s) + sum(bwl_s)) / (max(finish_s) - item['task_recode']['start_time'])  # min(start_s)
            result += r1

            # r1 /= len(item['task_recode']['task_list'])
            # sla_de = sum(slat_s) / len(
            #     item['task_recode']['task_list'])  #/ (max(finish_s) - item['task_recode']['start_time'])
            # if sla_de > 100000000:
            #     print('????')
            # r1 /= 2000
            # r2 = (sum(expend_s)) / (sum(mipsl_s)+sum(bwl_s))
            # r2 = 1/r2
            # if np.isnan(sla_de):
            #     print('????',np.isnan(sla_de))
            # # r2 *= (1-sla_de)
            # r2 /= 100
            # r2 *= 0.8
            # item['task_recode']['r'] = (Parameters.r1_rate * r1 + Parameters.r2_rate * r2) * (
            #     max(1 - sla_de, 0.1)) - sla_de
    return result

def dcR1Score(dc):
    cluster_list = dc.cluster_list
    score = 0
    for clu in cluster_list:
        score += cluR1Score(clu)
    return score

def handle_track(tracks):
    l = len(tracks)
    if l < 2:
        return False

    s = tracks[l-1]['s']
    a = tracks[l-1]['a']
    r = tracks[l-1]['r']
    s_prime = tracks[l-1]['s']
    done = 1
    i = l - 2
    total_r = r
    gae_r = r
    total_gae_r = gae_r
    while i >= 0:
        s = tracks[i]['s']
        a = tracks[i]['a']
        r = tracks[i]['r']
        s_prime = tracks[i+1]['s']
        done = 0
        i -= 1

        total_r += r
        gae_r = gae_r*DRLParameters.gamma + r
        total_gae_r += gae_r
    return total_r, total_gae_r, l

def dcLoadBalnace(dc):
    result = np.zeros((len(dc.cluster_list), 2))
    for index, clu in enumerate(dc.cluster_list):
        clu_result = cluLoadBalance(clu)
        result[index][0] = clu_result.mean()
        result[index][1] = clu_result.std()
    return result

def cluLoadBalance(clu):
    result = np.zeros(len(clu.machine_list))
    for index, mc in enumerate(clu.machine_list):
        if len(mc.over_task_list) > 0:
            for task in mc.over_task_list:
                result[index] += (task.finish_time - task.start_time)
    return result