# -*- coding: utf-8 -*-
"""
Created on Thu May 17 15:45:10 2018

@author: wangshuang
"""


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''        一、加载包             '''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from scipy.stats import mode
import sys
'''根据实际情况调整路径，防止perform模块无法读取'''
sys.path.append('C:\\Users\\wangshuang\\Documents\\Python Scripts') 
import perform as pf #载入自定义函数
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

from lightgbm.sklearn import LGBMRegressor
from sklearn.grid_search import GridSearchCV

import datetime


#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''        二、导入数据            '''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''根据具体数据情况提取入模的X和Y，从系统中导出的数据需删除第二行中文注释，否则会影响后续对数据类型的判断'''
'''另外，对于法院被执行人字段，入模前数据需删除被执行人的具体信息变量，只保留flag即可（否则会将被执行人信息当做类别变量处理，出现错误）'''

'''datapath为变量数据的导入路径'''
datapath='C:/Users/wangshuang/Desktop/工作文档/模型数据/海通/海通数据.csv'


f=open(datapath,encoding='UTF-8')
Data=pd.read_csv(f)
Data.index=Data.iloc[:,0]
f.close()
Y=Data.iloc[:,1] #注意，此处1代表坏客户，0代表好客户
X=Data.iloc[:,2:-7] 



#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     三、数据预处理part1 删掉缺失率过高的变量,删掉同值过高的变量（可根据实际情况不操作） '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


''' 1.粗筛变量，删掉缺失值超过nan_ratio_threshold的变量 '''

nan_ratio_threshold=0.95 #nan_ratio_threshold为阈值

count_null=np.zeros(np.shape(X))
count_null[np.where(X.isnull())]=1
count_null_sumfactor=sum(count_null)/np.shape(X)[0]
X=X.iloc[:,np.where(count_null_sumfactor<=nan_ratio_threshold)[0]] 


''' 2.删掉非nan同值超过mode_ratio_threshold的变量 '''

mode_ratio_threshold=0.95 #mode_ratio_threshold为阈值

raw_feature_num=len(X.columns)
if_delete_feature=np.zeros([raw_feature_num,1])
for i in range(0,raw_feature_num):
    if_delete_feature[i]=(len(np.where(X.iloc[:,i]==mode(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])[0][0])[0])/len(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])>mode_ratio_threshold)
X=X.iloc[:,np.where(if_delete_feature==0)[0]]



#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     四、数据预处理part2 填补空缺值：数值类型变量填充值为-99，类别型变量填充值为blank  '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


X_types=pd.Series(X.dtypes,dtype='str')

for i in range(0,np.shape(X)[1]):
    if len(np.where(X.iloc[:,i].isna())[0])>0: #若有缺失值
        if X_types[i]=='float64' or X_types[i]=='int64':#若为数值型，则填充为-99           
            X.iloc[np.where(X.iloc[:,i].isna())[0],i]=-99                   
        else:#若为分类型，则填充为blank
            X.iloc[np.where(X.iloc[:,i].isna())[0],i]='blank'




#%%
            
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''           五、类别变量处理--转化为哑变量  (有类别变量的情况下运行）             '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''              
            
X_num_position=np.where((X_types=='int64')|(X_types=='float64'))[0]
X_num=X.iloc[:,X_num_position]
X_num_types=X_types[X_num_position]

X_category_position=np.where((X_types=='object')|(X_types=='str'))[0]
X_category=X.iloc[:,X_category_position]
X_category_types=X_types[X_category_position]

X_category_dummies=pd.get_dummies(X_category) #将类型变量变换为哑变量
X_category_dummies_belongs=np.zeros([len(X_category_dummies.columns),1])

#找到哑变量和原始变量的对应关系
for i in range(0,len(X_category_dummies.columns)):
    for j in range(0,len(X_category.columns)):
        if X_category_dummies.columns[i].find(X_category.columns[j])>=0:
            X_category_dummies_belongs[i]=j
            break            
            
X_dummies=X_num.join(X_category_dummies)         
            

X=X_dummies


#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''           六、对训练集和测试集做划分             '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''            

'''设定训练集和测试集比例，可调'''            
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=7)            

X=x_train
Y=y_train





#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     七、处理样本不均衡问题 1. 对多样本欠抽样  （可选）         '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''

massive_sample_position=np.where(Y==0)[0]

random.seed(7)
keep_massive_sample_position=random.sample(list(massive_sample_position),int(len(massive_sample_position)/2))

little_sample_position=list(np.where(Y==1)[0])

keep_sample_position=keep_massive_sample_position+little_sample_position


X=X.iloc[keep_sample_position,:]
Y=Y.iloc[keep_sample_position]

'''


#%%



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     八、处理样本不均衡问题  2.使用重复抽样（原小样本+有放回复制小样本）方式处理样本不均衡问题  '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

little_sample_target=0.1#目标小样本比例（可调整）

    
'''先根据小样本占比判断是否需要重复抽样：判断小样本占总样本比例是否低于little_sample_target'''
if min(len(np.where(Y==0)[0])/len(Y),len(np.where(Y==1)[0])/len(Y))<little_sample_target:
    
    little_sample_target_num=round(little_sample_target*X.shape[0])
    little_sample=X.iloc[np.where(Y==1)[0],:]
    little_sample_num=little_sample.shape[0]
    
    need_new_little_num=little_sample_target_num-little_sample_num
    
    np.random.seed(7)#设定random seed 保证每次跑不受随机因素影响
    random_list=np.floor(np.random.rand(need_new_little_num,1)*little_sample_num) #生成有放回抽样的随机数序列
    random_list=np.array(random_list,dtype='int64')
    
    X_new=little_sample.iloc[list(random_list[:,0]),:]  
    X=X.append(X_new,ignore_index=True)
     
    Y=Y.append(pd.Series(np.ones([need_new_little_num])),ignore_index=True)

#解决不运行上述代码时的index问题
Y.index=list(range(0,len(Y.index)))
X.index=list(range(0,len(X.index)))



#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     九、模型调参    1.计算最初的feature_importance  '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''先给定一组常见的参数组合，计算变量的feature_importance'''
lgbm_model = LGBMRegressor(
learning_rate =0.05,
n_estimators=500,
max_depth=5,
min_child_weight=0.0005,
min_child_samples=3,
min_split_gain=0.1,
subsample=0.8,
colsample_bytree=0.8,
objective= 'binary',
random_state=7)

lgbm_param = lgbm_model.get_params()
lgbm_train = lgb.Dataset(X,Y)

'''使用交叉验证的方式确定最优的树数量'''
cvresult = lgb.cv(lgbm_param, lgbm_train, num_boost_round=lgbm_param['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
best_n_estimators=len(cvresult['auc-mean'])

lgbm_model.set_params(n_estimators=best_n_estimators)
lgbm_model.fit(X,Y,eval_metric='auc')
feat_imp = pd.Series(lgbm_model.feature_importances_,index=X.columns)   
feat_imp=feat_imp.sort_values(ascending=False)


valid_feature_num=len(np.where(feat_imp>0)[0]) #有效变量是有feature_importance的变量（在lgbm树模型中有贡献的变量，其他的变量没有用到）


#%%


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     九、模型调参    2.调优feature_num          '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''feature_num的取值范围根据实际valid_feature_num取值来定'''
feature_num_range=[50,70,100,200,valid_feature_num]

ks_auc=pd.DataFrame()
for feature_num in feature_num_range: #计算在不同的feature_num下xgb模型在测试集的KS和AUC表现
    
    chosen_feature=feat_imp.index[:feature_num] #选取feature_importance排在前feature_num的变量
    
    lgbm_model.set_params(n_estimators=500)
    lgbm_param_temp = lgbm_model.get_params()
    
    lgbm_train = lgb.Dataset(X.loc[:,chosen_feature],Y)
    
    cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=lgbm_param_temp['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
    best_n_estimators_temp=len(cvresult['auc-mean'])
    
    lgbm_model.set_params(n_estimators=best_n_estimators_temp)
    lgbm_model.fit(X.loc[:,chosen_feature],Y,eval_metric='auc')   
    preds=lgbm_model.predict(x_test.loc[:,chosen_feature]) 
    
    ks_value,bad_percent,good_percent=pf.cal_ks(-preds,y_test,section_num=20)
                      
    false_positive_rate,recall,thresholds = roc_curve(y_test, preds)
    roc_auc=auc(false_positive_rate,recall) 
    
    ks_auc=pd.concat([ks_auc,pd.DataFrame([np.max(ks_value),roc_auc]).T])
    
ks_auc.columns=['ks','auc']
ks_auc.index=feature_num_range
    
print(ks_auc)    


'''final_feature_num可以选取KS和AUC相加最高的，也可以手动指定'''
ks_add_auc=ks_auc.iloc[:,0]+ks_auc.iloc[:,1]

final_feature_num=feature_num_range[int(np.where(ks_add_auc==np.max(ks_add_auc))[0])]
#final_feature_num=200
print('入模变量数量为{0}时，KS和AUC最高'.format(final_feature_num))


chosen_final_feature=feat_imp.index[:final_feature_num]

'''选好feature_num后重新矫正best_n_estimators'''
lgbm_model.set_params(n_estimators=500)
lgbm_param_temp = lgbm_model.get_params()
lgbm_train = lgb.Dataset(X.loc[:,chosen_final_feature],Y)
cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=lgbm_param_temp['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
best_n_estimators=len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)


#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     九、模型调参    3.调优max_depth,min_child_samples, min_split_gain, 确定xgb整体架构    '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

param_test1 = {
'max_depth':list(range(3,7)),
'min_child_samples':[1,3,5,10],
'min_split_gain':[0,0.1,0.2,0.3,0.4,0.5,0.7,1,2],
}

gsearch1 = GridSearchCV(lgbm_model,param_grid=param_test1,scoring='roc_auc',cv=5)

starttime = datetime.datetime.now() 
gsearch1.fit(X.loc[:,chosen_final_feature],Y)
endtime = datetime.datetime.now()
print('第一次gridsearch耗时{0} seconds'.format((endtime - starttime).seconds))

gsearch1.grid_scores_,gsearch1.best_params_,gsearch1.best_score_


'''可以直接读取最大auc对应的参数设置，也可以根据观察和经验选取效果好且稳定的参数组合'''
#lgbm_model.set_params(max_depth=gsearch1.best_params_['max_depth'])
#lgbm_model.set_params(min_child_samples=gsearch1.best_params_['min_child_samples'])
#lgbm_model.set_params(min_split_gain=gsearch1.best_params_['min_split_gain'])

lgbm_model.set_params(max_depth=5)
lgbm_model.set_params(min_child_samples=1)
lgbm_model.set_params(min_split_gain=0.3)


'''选好参数后重新矫正best_n_estimators'''
lgbm_model.set_params(n_estimators=500)
lgbm_param_temp = lgbm_model.get_params()
lgbm_train = lgb.Dataset(X.loc[:,chosen_final_feature],Y)
cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=lgbm_param_temp['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
best_n_estimators=len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)



#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     九、模型调参    4.调优subsample , colsample_bytree    '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

param_test2 = {
'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)]
}

gsearch2 = GridSearchCV(lgbm_model,param_grid=param_test2,scoring='roc_auc',cv=5)

starttime = datetime.datetime.now() 
gsearch2.fit(X.loc[:,chosen_final_feature],Y)
endtime = datetime.datetime.now()
print('第二次gridsearch耗时{0} seconds'.format((endtime - starttime).seconds))

gsearch2.grid_scores_,gsearch2.best_params_,gsearch2.best_score_


'''可以直接读取最大auc对应的参数设置，也可以根据观察和经验选取效果好且稳定的参数组合'''
#lgbm_model.set_params(subsample=gsearch2.best_params_['subsample'])
#lgbm_model.set_params(colsample_bytree=gsearch2.best_params_['colsample_bytree'])

lgbm_model.set_params(subsample=0.8)
lgbm_model.set_params(colsample_bytree=0.8)


'''选好参数后重新矫正best_n_estimators'''
lgbm_model.set_params(n_estimators=500)
lgbm_param_temp = lgbm_model.get_params()
lgbm_train = lgb.Dataset(X.loc[:,chosen_final_feature],Y)
cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=lgbm_param_temp['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
best_n_estimators=len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)




#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     九、模型调参    5.调优正则项参数reg_alpha , reg_lambda    '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

param_test3 = {
'reg_alpha':[0,0.2,0.5,1,2],
'reg_lambda':[0,0.5,1,2,5,10]
}

gsearch3 = GridSearchCV(lgbm_model,param_grid=param_test3,scoring='roc_auc',cv=5)

starttime = datetime.datetime.now() 
gsearch3.fit(X.loc[:,chosen_final_feature],Y)
endtime = datetime.datetime.now()
print('第三次gridsearch耗时{0} seconds'.format((endtime - starttime).seconds))

gsearch3.grid_scores_,gsearch3.best_params_,gsearch3.best_score_


'''可以直接读取最大auc对应的参数设置，也可以根据观察和经验选取效果好且稳定的参数组合'''
lgbm_model.set_params(reg_alpha=gsearch3.best_params_['reg_alpha'])
lgbm_model.set_params(reg_lambda=gsearch3.best_params_['reg_lambda'])

#lgbm_model.set_params(reg_alpha=)
#lgbm_model.set_params(reg_lambda=)


'''选好参数后重新矫正best_n_estimators'''
lgbm_model.set_params(n_estimators=500)
lgbm_param_temp = lgbm_model.get_params()
lgbm_train = lgb.Dataset(X.loc[:,chosen_final_feature],Y)
cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=lgbm_param_temp['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
best_n_estimators=len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)


#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     九、模型调参    6.调优学习率learning_rate    '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

param_test4 = {
'learning_rate':[0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2],
}

gsearch4 = GridSearchCV(lgbm_model,param_grid=param_test4,scoring='roc_auc',cv=5)

starttime = datetime.datetime.now() 
gsearch4.fit(X.loc[:,chosen_final_feature],Y)
endtime = datetime.datetime.now()
print('第四次gridsearch耗时{0} seconds'.format((endtime - starttime).seconds))

gsearch4.grid_scores_,gsearch4.best_params_,gsearch4.best_score_


'''可以直接读取最大auc对应的参数设置，也可以根据观察和经验选取效果好且稳定的参数组合'''
lgbm_model.set_params(learning_rate=gsearch4.best_params_['learning_rate'])

#lgbm_model.set_params(learning_rate=)


'''选好参数后重新矫正best_n_estimators'''
lgbm_model.set_params(n_estimators=500)
lgbm_param_temp = lgbm_model.get_params()
lgbm_train = lgb.Dataset(X.loc[:,chosen_final_feature],Y)
cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=lgbm_param_temp['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
best_n_estimators=len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)


print('xgb模型调参完成！最终参数：')
print(lgbm_model.get_params())



#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''          十、选择参数后，生成light-gbm model         '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

lgbm_model.fit(X.loc[:,chosen_final_feature],Y,eval_metric='auc')

preds = lgbm_model.predict(x_test.loc[:,chosen_final_feature]) 

ks_value,bad_percent,good_percent=pf.cal_ks(-preds,y_test,section_num=20)                    
max_ks0=np.max(ks_value)
                
false_positive_rate, recall, thresholds = roc_curve(y_test, preds)
roc_auc0=auc(false_positive_rate,recall) 

print('当前模型在样本内测试集的KS值和AUC值分别为{0}'.format([max_ks0,roc_auc0]))

plt.figure()
plt.hist(preds)
plt.ylabel('Number of samples')
plt.xlabel('probability of y=1')
plt.title('Probability Distribution on test samples')

plt.figure()
plt.plot(list(range(0,21)),np.append([0],bad_percent),'-r',label='Bad Percent')
plt.plot(list(range(0,21)),np.append([0],good_percent),'-g',label='Good Percent')
plt.plot(list(range(0,21)),np.append([0],ks_value),'-b',label='KS value')
plt.legend(loc='lower right')
plt.ylabel('% of total Good/Bad')
plt.xlabel('% of population')

plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc0)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()



#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''          十三、保存 model 和选取的feature names           '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''output_file_path是输出lgbm模型文件和所选变量的文件夹路径，按照实际情况调整'''
output_file_path='C:/Users/wangshuang/Desktop'


lgbm_model.booster_.save_model(output_file_path+'/lgbm.model') # 用于存储训练出的模型
chosen_feature_pd=pd.DataFrame(chosen_final_feature)
chosen_feature_pd.to_csv(output_file_path+'/lgbm_features.csv',encoding="utf_8_sig",index=False)



#%%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''       十四、读取保存好的 model 和feature names   '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''input_model_path是导入之前保存好的lgbm model的路径，input_xgb_features_path是导入lgbm model对应的features的csv文件路径 ，按照实际情况调整 '''
input_model_path='C:/Users/wangshuang/Desktop/lgbm.model'
input_xgb_features_path='C:/Users/wangshuang/Desktop/lgbm_features.csv'


lgbm_model = lgb.Booster(model_file=input_model_path)
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))


preds = lgbm_model.predict(x_test.loc[:,chosen_feature]) 

ks_value,bad_percent,good_percent=pf.cal_ks(-preds,y_test,section_num=20)                    
max_ks0=np.max(ks_value)
                
false_positive_rate, recall, thresholds = roc_curve(y_test, preds)
roc_auc0=auc(false_positive_rate,recall) 

print('当前模型在样本内测试集的KS值和AUC值分别为{0}'.format([max_ks0,roc_auc0]))

plt.figure()
plt.hist(preds)
plt.ylabel('Number of samples')
plt.xlabel('probability of y=1')
plt.title('Probability Distribution on test samples')

plt.figure()
plt.plot(list(range(0,21)),np.append([0],bad_percent),'-r',label='Bad Percent')
plt.plot(list(range(0,21)),np.append([0],good_percent),'-g',label='Good Percent')
plt.plot(list(range(0,21)),np.append([0],ks_value),'-b',label='KS value')
plt.legend(loc='lower right')
plt.ylabel('% of total Good/Bad')
plt.xlabel('% of population')

plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc0)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()


