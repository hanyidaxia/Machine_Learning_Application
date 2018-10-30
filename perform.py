# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:52:35 2018

@author: wangshuang
"""
import pandas as pd
import numpy as np

def cal_ks(point,Y,section_num=20):
    Y=pd.Series(Y)
    sample_num=len(Y)
    
    bad_percent=np.zeros([section_num,1])
    good_percent=np.zeros([section_num,1])

    
    point=pd.DataFrame(point)
    sorted_point=point.sort_values(by=0)
    total_bad_num=len(np.where(Y==1)[0])
    total_good_num=len(np.where(Y==0)[0])
    
    for i in range(0,section_num):
        split_point=sorted_point.iloc[int(round(sample_num*(i+1)/section_num))-1]
        position_in_this_section=np.where(point<=split_point)[0]
        bad_percent[i]=len(np.where(Y.iloc[position_in_this_section]==1)[0])/total_bad_num
        good_percent[i]=len(np.where(Y.iloc[position_in_this_section]==0)[0])/total_good_num
        
    ks_value=bad_percent-good_percent

    return ks_value,bad_percent,good_percent



def WOE(data, dim, bucket_num=10, auto=True, bond_num=[]):
    m = data.shape[0]
    X = data[:, dim]
    y = data[:, -1]
    tot_bad = np.sum(y == 1)
    tot_good = np.sum(y == 0)
    data = np.column_stack((X.reshape(m, 1), y.reshape(m, 1)))
    cnt_bad = []
    cnt_good = []
    min = np.min(data[:, 0])
    max = np.max(data[:, 0])
    if auto == True:

        index = np.linspace(min, max, bucket_num + 1)
    else:
        index = bond_num
        bucket_num = bond_num.shape[0] - 1
    data_bad = data[data[:, 1] == 1, 0]
    data_good = data[data[:, 1] == 0, 0]
    eps = 1e-8
    for i in range(bucket_num):
        if i < bucket_num - 1:
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad > index[i] - eps, data_bad < index[i + 1])))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good > index[i] - eps, data_good < index[i + 1])))
        else:
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad > index[i] - eps, data_bad < index[i + 1] + eps)))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good > index[i] - eps, data_good < index[i + 1] + eps)))
    bond = np.array(index)
    cnt_bad = np.array(cnt_bad)
    cnt_good = np.array(cnt_good)
    
    #对完美分箱增加一个虚拟样本，保证有woe值
    cnt_bad[cnt_bad==0]+=1
    cnt_good[cnt_good==0]+=1
    
    
    length = cnt_bad.shape[0]
    for i in range(length):
        j = length - i - 1
        if j != 0:
            if cnt_bad[j] == 0 or cnt_good[j] == 0:
                cnt_bad[j - 1] += cnt_bad[j]
                cnt_good[j - 1] += cnt_good[j]
                cnt_bad = np.append(cnt_bad[:j], cnt_bad[j + 1:])
                cnt_good = np.append(cnt_good[:j], cnt_good[j + 1:])
                bond = np.append(bond[:j], bond[j + 1:])
    if cnt_bad[0] == 0 or cnt_good[0] == 0:
        cnt_bad[1] += cnt_bad[0]
        cnt_good[1] += cnt_good[0]
        cnt_bad = cnt_bad[1:]
        cnt_good = cnt_good[1:]
        bond = np.append(bond[0], bond[2:])
    woe = np.log((cnt_bad / tot_bad) / (cnt_good / tot_good))
    IV = ((cnt_bad / tot_bad) - (cnt_good / tot_good)) * woe
    IV_tot = np.sum(IV)
    bond_str = []
    for b in bond:
        bond_str.append(str(b))
    box_num=  cnt_bad+ cnt_good
    bad_rate=cnt_bad/box_num
    
    return IV_tot, IV, woe, bond, box_num, bad_rate


def PSI(score_train,score_test,section_num=10):
    score_train=pd.DataFrame(score_train)
    score_test=pd.DataFrame(score_test)
    
    total_train_num=len(score_train)
    total_test_num=len(score_test)
    
    sorted_score_train=score_train.sort_values(by=0)    
    
    PSI_value=0
    
    for i in range(0,section_num):
        lower_bound=sorted_score_train.iloc[int(round(total_train_num*(i)/section_num))]
        higher_bound=sorted_score_train.iloc[int(round(total_train_num*(i+1)/section_num))-1]
        score_train_percent=len(np.where((score_train>=lower_bound)&(score_train<=higher_bound))[0])/total_train_num
        score_test_percent=len(np.where((score_test>=lower_bound)&(score_test<=higher_bound))[0])/total_test_num
        
        PSI_value+=(score_test_percent-score_train_percent)*np.log(score_test_percent/score_train_percent)
        
    return PSI_value
    
    
    

