from Base.Parameters import Parameters
import numpy as np

def reward4Cluster_LoadBalance(cluster):
    '''
    集群的回报组成：手底下mc的负载状态 和 负载均衡程度，任务的超时时长占本身时长占比
    :param cluster:
    :return:
    '''
    total_source_p = {}  # 每台机器对应的可用资源百分比
    for s in Parameters.BaseSource:
        total_source_p[s + '_p'] = []
    # a_mc_source = {}  # 统计mc的资源（可用，_p可用百分比， _total总资源）
    # 对集群资源进行统计 生成a_mc_source & total_a_source
    for mc in cluster.machine_list:
        # a_mc_source[mc] = {}
        for s in Parameters.BaseSource:
            # a_mc_source[mc][s] = mc.a_source[s]
            total_source_p[s + '_p'].append(mc.a_source[s] / mc.source[s])
            # a_mc_source[mc][s + '_p'] = mc.a_source[s] / mc.source[s]
            # a_mc_source[mc][s + '_total'] = mc.source[s]
        # a_mc_source[mc]['task'] = []
    result = 0
    for s in Parameters.BaseSource:
        total_source_p[s + '_std'] = np.array(total_source_p[s + '_p']).std()  # 集群内各项资源对应的标准差
        total_source_p[s + '_mean'] = np.array(total_source_p[s + '_p']).mean()
        result += total_source_p[s + '_std']
    result /= len(Parameters.BaseSource)
    return result

def advantageReward4Cluster_Balance(cluster, task, mc_right):
    total_source_p = {}  # 每台机器对应的可用资源百分比
    for s in Parameters.BaseSource:
        total_source_p[s + '_p'] = []
    a_mc_source = {}  # 统计mc的资源（可用，_p可用百分比， _total总资源）
    # 对集群资源进行统计 生成a_mc_source & total_a_source
    i = 0
    for mc in cluster.machine_list:
        a_mc_source[mc] = {}
        a_mc_source[mc]['ed'] = False
        a_mc_source[mc]['index'] = i
        i += 1
        for s in Parameters.BaseSource:
            a_mc_source[mc][s] = mc.a_source[s]
            if mc.matchTask(task):
                a_mc_source[mc][s+'_ed'] = mc.a_source[s]-task.need_source[s]
                a_mc_source[mc]['ed'] = True
            total_source_p[s + '_p'].append(mc.a_source[s] / mc.source[s])
            # a_mc_source[mc][s + '_p'] = mc.a_source[s] / mc.source[s]
            # a_mc_source[mc][s + '_total'] = mc.source[s]
    result_root = 0
    for s in Parameters.BaseSource:
        total_source_p[s + '_std'] = np.array(total_source_p[s + '_p']).std()  # 集群内各项资源对应的标准差
        total_source_p[s + '_mean'] = np.array(total_source_p[s + '_p']).mean()
        result_root += total_source_p[s + '_std']
    result_root /= len(Parameters.BaseSource)
    result_list = []
    result_right = None
    for mc, data in a_mc_source.items():
        if data['ed']:
            tmp_result = 0
            for s in Parameters.BaseSource:
                tmp_s_p = data[s+'_ed']/mc.source[s]
                recode_total_source_p = total_source_p[s + '_p'][data['index']]
                total_source_p[s + '_p'][data['index']] = tmp_s_p
                s_std = np.array(total_source_p[s + '_p']).std()
                total_source_p[s + '_p'][data['index']] = recode_total_source_p
                tmp_result += s_std
            tmp_result /= len(Parameters.BaseSource)
            result_list.append(result_root-tmp_result)
            if mc == mc_right:
                result_right = result_root-tmp_result
    result_mean = np.array(result_list).mean()
    # return result_list

    '''
    
    if result_right == None:
        return -1.6663520, result_list
    try:
        if result_right - np.array(result_list).max() == 0:
            return 1, result_list
    except TypeError as e:
        print(e)
    return 0, result_list
    
    '''
    return result_right-result_mean, result_list
    # return min(result_right-result_mean, 1), result_list
    # return max(result_right-result_mean, 0), result_list
    if result_right > result_mean: #or (result_mean-result_right)>0.01
        return 1, result_list
    # # result_min = np.array(result_list).min()
    # # if result_right == result_min:
    # #     return 1
    # # if (result_mean-result_right)>0.01:
    # #     return 1
    return 0, result_list

def reward4Cluster_ResourceUtilization(cluster):
    total_source_p = {}  # 每台机器对应的可用资源百分比
    for s in Parameters.BaseSource:
        total_source_p[s + '_u'] = []
        total_source_p[s + '_all'] = []
        # total_source_p[s + '_p'] = []
    # a_mc_source = {}  # 统计mc的资源（可用，_p可用百分比， _total总资源）
    # 对集群资源进行统计 生成a_mc_source & total_a_source
    for mc in cluster.machine_list:
        # a_mc_source[mc] = {}
        for s in Parameters.BaseSource:
            total_source_p[s + '_u'].append(mc.source[s] - mc.a_source[s])
            total_source_p[s + '_all'].append(mc.source[s])
    result = 0
    for s in Parameters.BaseSource:
        result += (np.array(total_source_p[s + '_u']).sum()/np.array(total_source_p[s + '_all']).sum())
    return result/len(Parameters.BaseSource)

def reward4Task_ExTime(task):
    try:
        return (task.start_time - task.load_time - 1)/task.duration
    except ZeroDivisionError as e:
        print(e)

def extractRewardForRecode(start, dur, recodes):
    recodes_l = len(recodes)
    i = 0
    while i < recodes_l:
        item=recodes[i]
        if start < item['time']:
            i -= 1
            break
        i += 1
    reward = 0
    c_start = start
    while i < recodes_l-1:
        item = recodes[i]
        item_n = recodes[i+1]
        if item['time'] >= start + dur:
            break
        if start + dur >= item_n['time']:
            reward += ((item['balance']/item['mc_num'])*(item_n['time']-c_start))
            c_start = item_n['time']
        else:
            reward += ((item['balance'] / item['mc_num']) * (start + dur - item['time']))
            c_start = start + dur
        i += 1
    return -reward

def reviseTrack(cluster):
    tracks = cluster.drl_track
    # recodes = cluster.recode_list
    for item in tracks:
        if item['task_recode']['r'] == None:
            start_s = []
            finish_s = []
            mipsl_s = []
            bwl_s = []
            slat_s = []
            for task in item['task_recode']['task_list']:
                start_s.append(task.start_time)
                finish_s.append(task.finish_time)
                mipsl_s.append(task.mips_length)
                bwl_s.append(task.bw_length)
                slat_s.append(task.start_time-task.load_time)
            r = (sum(mipsl_s)+sum(bwl_s))/(max(finish_s) - item['task_recode']['start_time']) # min(start_s)
            r /= len(item['task_recode']['task_list'])
            sla_de = sum(slat_s)/(max(finish_s) - item['task_recode']['start_time'])/len(item['task_recode']['task_list'])
            if sla_de > 1:
                print('????')
            r *= (1-sla_de)
            r /= 2000
            item['task_recode']['r'] = r
        item['r'] = item['task_recode']['r']

def reviseTrackWithExpend(cluster):
    tracks = cluster.drl_track
    # recodes = cluster.recode_list
    for item in tracks:
        if item['task_recode']['r'] == None:
            start_s = []
            finish_s = []
            mipsl_s = []
            bwl_s = []
            slat_s = []
            expend_s = []
            slarate_s = []
            for task in item['task_recode']['task_list']:
                start_s.append(task.start_time)
                finish_s.append(task.finish_time)
                mipsl_s.append(task.mips_length)
                bwl_s.append(task.bw_length)
                slat_s.append(task.start_time-task.load_time)
                slarate_s.append((task.start_time-task.load_time)/(task.finish_time-task.load_time))
                expend_s.append(task.expend)
            # r = (sum(mipsl_s)+sum(bwl_s))/(max(finish_s) - item['task_recode']['start_time']) # min(start_s)
            r = (sum(mipsl_s)+sum(bwl_s))/(sum(expend_s))
            r /= len(item['task_recode']['task_list'])
            sla_de = sum(slat_s)/(max(finish_s) - item['task_recode']['start_time'])/len(item['task_recode']['task_list'])
            sla_de = np.sqrt(sla_de)
            # sla_de = np.array(slarate_s).mean()
            # print(sla_de)

            if sla_de > 1 or np.isnan(sla_de):
                print('????')
            r *= (1-sla_de)
            r /= 50
            item['task_recode']['r'] = r
        item['r'] = item['task_recode']['r']


def reviseTrackWithExpendAndMakespan(cluster):
    tracks = cluster.drl_track
    # recodes = cluster.recode_list
    result = 0
    for item in tracks:
        if item['task_recode']['r'] == None:
            start_s = []
            finish_s = []
            mipsl_s = []
            bwl_s = []
            slat_s = []
            expend_s = []
            slarate_s = []
            sla_abs_s = []
            for task in item['task_recode']['task_list']:
                start_s.append(task.start_time)
                finish_s.append(task.finish_time)
                mipsl_s.append(task.mips_length)
                bwl_s.append(task.bw_length)
                # slat_s.append(task.start_time-task.load_time)
                sla_abs_s.append(task.finish_time - task.load_time - task.duration)
                slat_s.append(max(task.finish_time - task.load_time - task.duration, 0))
                slarate_s.append((task.start_time-task.load_time)/(task.finish_time-task.load_time))
                expend_s.append(task.expend)

            r1 = (sum(mipsl_s) + sum(bwl_s)) / (max(finish_s) - item['task_recode']['start_time'])  # min(start_s)
            r1 /= len(item['task_recode']['task_list'])
            sla_de = sum(slat_s) / len(
                item['task_recode']['task_list'])  #/ (max(finish_s) - item['task_recode']['start_time'])
            if sla_de > 100000000:
                print('????')
            # r1 *= (1 - sla_de**(1/20))
            r1 /= 2000

            r2 = (sum(expend_s)) / (sum(mipsl_s)+sum(bwl_s))
            r2 /= len(item['task_recode']['task_list'])
            r2 = 1/r2
            # sla_de = np.sqrt(sla_de)
            # sla_de = np.array(slarate_s).mean()
            # print(sla_de)

            if np.isnan(sla_de):
                print('????',np.isnan(sla_de))
            # r2 *= (1-sla_de)
            r2 /= 500
            r2 *= 0.8
            # r2 /= 400
            # if sla_de == 0:
            #     # item['task_recode']['r'] = (max(sla_de - 1, 0.1) * r1 + max(1 - sla_de, 0.1) * r2) * (
            #     #     max(1 - sla_de, 0.1)) - sla_de
            #     item['task_recode']['r'] = (Parameters.r1_rate * r1 + Parameters.r2_rate * r2) * (
            #         max(1 - sla_de, 0.01)) - sla_de
            # if sla_de < 100:
            #     item['task_recode']['r'] = (Parameters.r1_rate*r1 + Parameters.r2_rate*r2)*(max(1-sla_de, 0.01)) - sla_de
            # else:
            #     item['task_recode']['r'] = (Parameters.r1_rate * r1 + Parameters.r2_rate * r2) * (
            #             max(1 - sla_de, 0.1)) - 0.1*sla_de
            # if sla_de < 1:
            #     print(sum(sla_abs_s)/len(item['task_recode']['task_list']))
            # sla_de = 0
            # item['task_recode']['r'] = (Parameters.r1_rate * r1 + Parameters.r2_rate * r2) * (
            #     max(1 - sla_de, 0.1)) - sla_de
            item['task_recode']['r'] = r1
            # item['task_recode']['r'] *= (1 - sla_de ** (1 / 10000))
            # print(r1,r2, sla_de)
            result += (r1*len(item['task_recode']['task_list']))
        item['r'] = item['task_recode']['r']
        ot1 = item['task'].finish_time - item['task'].load_time - item['task'].duration
        ot = max(ot1, 0)  # item['task_recode']['sla_de'] #max(ot1,0)
        item['r'] = (Parameters.r1_rate *item['task_recode']['r'] + Parameters.r2_rate * (item['task'].mips_length+item['task'].bw_length)*0.8/item['task'].expend/100)*(
                max(1 - ot, 0.1)) - ot
    return result

# def reviseTrack(cluster):
#     tracks = cluster.drl_track
#     recodes = cluster.recode_list
#     for item in tracks:
#         item['r'] =

def reward4Cluster_Makespan(wait, dut, ext, plant):
    # print(dut, ext, ext/dut)
    r0 =  (1 / (dut))
    r1 = (1 / (dut)) * max((1 - ext / dut), 0.1)
    if r1 < 0:
        print(123)
    r2 = (1 / (dut + wait)) * (1 - ext / dut)
    r3 = (1 / (dut)) * (1 - ext / plant)
    r4 = (1 / (dut + wait)) * (1 - ext / plant)
    return r1*10