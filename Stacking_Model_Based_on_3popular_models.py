
# coding: utf-8

# # 基于XGB，LGB, Catb三种模型变量综合之后的stacking新模型1.0

# ### 为了获取直观的变量预测能力对比，先使用逻辑回归模型对变量进行训练测试划分,之后使用统一训练集数据

# #### 1.导入数据

# In[1]:


# 载入程序包
import os
import pandas as pd
import numpy as np
import scorecardpy as sc
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from scipy.stats import mode
import sys
import re
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
'''根据实际情况调整路径，防止perform模块无法读取,perform模块'''
'''perform模块conda目前无法安装，只能本地导入'''
sys.path.append('C:\\Users\\Bairong\\Desktop\\light gbm') 
import perform as pf #载入自定义函数
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMRegressor
from sklearn.grid_search import GridSearchCV
import datetime
import seaborn as sns
import catboost as ct
from catboost import Pool, CatBoostClassifier, cv
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


# In[28]:


# 修改工作路径，以便读取数据
os.getcwd()
os.chdir("C:/Users/Bairong/Desktop")


# In[29]:


# 读取数据
def float_1(x):
    try:
        return float(x)
    except:
        return np.nan
data = pd.read_csv("SAMPLE3.csv")

# 只留下一个标签作为模型因变量，删除数据集中的灰客户
data1 = data[data["flagy"] != 2]    
print(data1.shape)


# #### 2.变量粗筛
#   
#   

# #### 删除某些明显不适合入模的变量

# In[30]:


drops = [ 'pc_business_type']
for item in drops:
    if item in data1.columns.values:
        del data1[item]


# In[31]:


X = data1.drop(['flagy'],1)
X.shape


# In[15]:


'''
    LGB 以及 XGboost模型都对类别型变量敏感，故法院被执行人字段在此次建模中全部删除。
    解释性不好的变量比如城市等级等变量在原始数据中改变为数值型变量，数值对应其等级，如果采用哑变量消除object数据值会极大增加变量个数，
    故除了本身与等级相关的变量可以采用哑变量处理，其他变量不建议使用哑变量。
'''
#删除模块变量的代码实现：
'''
import re
pat = "^ex|^sl|^other_"
for item in data1.columns.values.tolist():
    if re.search(pat, item) != None:
        del data1[item]
'''


# #### 变量粗筛方法
# 

变量粗筛有两种方法：
1、使用scorecard内置函数进行粗筛，R或Python实现。
>scorecard函数粗筛的内部筛选条件为：
        iv_limit	 The information value of kept variables should >= iv_limit. The default is 0.02.
        missing_limit	 The missing rate of kept variables should <= missing_limit. The default is 0.95.
        identical_limit	 The identical value rate (excluding NAs) of kept variables should <= identical_limit. The default is 0.95.
        var_rm	 Name of force removed variables, default is NULL.
        var_kp	 Name of force kept variables, default is NULL.
        return_rm_reason	 Logical, default is FALSE.
        positive	 Value of positive class, default is "bad|1".

手动粗筛的条件为:
        NA Ratio > 95%
        剩余非NAN变量同值率 > 95%
代码实现：
        1.粗筛变量，删掉缺失值超过nan_ratio_threshold的变量 

        nan_ratio_threshold=0.95 #nan_ratio_threshold为阈值

        count_null=np.zeros(np.shape(X))
        count_null[np.where(X.isnull())]=1
        count_null_sumfactor=sum(count_null)/np.shape(X)[0]
        X=X.iloc[:,np.where(count_null_sumfactor<=nan_ratio_threshold)[0]] 


        2.删掉非nan同值超过mode_ratio_threshold的变量 

        mode_ratio_threshold=0.95 #mode_ratio_threshold为阈值

        raw_feature_num=len(X.columns)
        if_delete_feature=np.zeros([raw_feature_num,1])
        for i in range(0,raw_feature_num):
            if_delete_feature[i]=(len(np.where(X.iloc[:,i]==mode(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])[0][0])[0])/len(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])>mode_ratio_threshold)
        X=X.iloc[:,np.where(if_delete_feature==0)[0]]


# #### 同一变量使用scorecard内置函数与手动筛选的结果比较

# In[32]:


#立信样本使用scorecard内置函数粗筛之后的效果
dt_s = sc.var_filter(data1, y = "flagy")
print(dt_s.shape)


# In[33]:


#立信样本使用
''' 1.粗筛变量，删掉缺失值超过nan_ratio_threshold的变量 '''

nan_ratio_threshold=0.95 #nan_ratio_threshold为阈值

count_null=np.zeros(np.shape(X))#计算空值量
count_null[np.where(X.isnull())]=1#计算非空值量并赋值为1
count_null_sumfactor=sum(count_null)/np.shape(X)[0]#计算变量空值占比
X=X.iloc[:,np.where(count_null_sumfactor<=nan_ratio_threshold)[0]] #取非空值小于95%的变量赋值给X


''' 2.删掉非nan同值超过mode_ratio_threshold的变量 '''

mode_ratio_threshold=0.95 #mode_ratio_threshold为阈值

raw_feature_num=len(X.columns)
if_delete_feature=np.zeros([raw_feature_num,1])
for i in range(0,raw_feature_num):
    if_delete_feature[i]=(len(np.where(X.iloc[:,i]==mode(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])[0][0])[0])/len(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])>mode_ratio_threshold)
X=X.iloc[:,np.where(if_delete_feature==0)[0]]

print(X.shape)
'''
第二步容易报错，因为在处理原始数据时，有的变量是int，obj混杂在一起的，执行以下代码：
mode(X.iloc[np.where(~X.iloc[:,i].isna())[0],i])
X.iloc[np.where(~X.iloc[:,i].isna())[0],i]
报错变量为混合类型数据，需要手动删除或强制转换

'''


# 根据筛选结果选择scorecard内置函数的粗筛结果作为粗筛数据
# 

# #### 3.对数据集缺失值进行处理

# In[34]:


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''     数据预处理part2 填补空缺值：数值类型变量填充值为-99，类别型变量填充值为blank  '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


dt_s_types=pd.Series(dt_s.dtypes,dtype='str')

for i in range(0,np.shape(dt_s)[1]):
    if len(np.where(dt_s.iloc[:,i].isna())[0])>0: #若有缺失
        if dt_s_types[i]=='float64' or dt_s_types[i]=='int64':#若为数值型，则填充为-99           
            dt_s.iloc[np.where(dt_s.iloc[:,i].isna())[0],i]=-99                   
        else:#若为分类型，则填充为blank
            dt_s.iloc[np.where(dt_s.iloc[:,i].isna())[0],i]='blank'




#%%


# #### 4.类别变量处理-哑变量 

# 在本数据集中所有类别型变量均已转变为数值型变量，故无需执行此步骤
# 若留有类别型变量，控制类别在五种之内，如果不止一个类别型变量且类别超过五种，则容易出现memory error

# In[ ]:


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''           五、类别变量处理--转化为哑变量  (有类别变量的情况下运行）             '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''           ###这一步需要判别数据集是否存在类别型变量，
                                                                               
    '''
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

####'''
#%%


# #### 5划分训练集和测试集 

# In[69]:


dt = dt_s.copy()  #最好用copy，否则会增加内存压力
y = dt_s['flagy']

col = set(dt_s.columns)
P = dt_s.ix[:, [x for x in col if not re.search('flagy|cus_num', x)]]
P = P.applymap(float_1)
P = P.astype(np.float)
#P = P.fillna(0)            #因为之前已经处理过空缺值并将其转变为-99，这一步无需执行
P_train, P_test, y_train, y_test = train_test_split(P, y, test_size=0.3, random_state=10)

dtrain = xgb.DMatrix(P_train, y_train)
dtest = xgb.DMatrix(P_test, y_test)
X = P_train
Y = y_train

'''P_train 和 P_test 作为之后三种模型使用的统一训练测试集'''


# ### 分别在三个模型中获取predction值
# 

# #### XGB模型
# 

# In[298]:


params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
        'eval_metric': 'auc',
    	'max_depth': 1,
    	'lambda': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.1, 
        'eta': 0.02,
    	'nthread': 8
        }

bst = xgb.train(params, dtrain)

num_round = 1000
bst = xgb.train(params, dtrain, num_round)

# 交叉验证选出test-auc-mean最高时对应的迭代次数
cv_log = xgb.cv(params,dtrain,num_boost_round=1000,
                nfold=5,
                metrics='auc',
                early_stopping_rounds=10)

bst_auc = cv_log['test-auc-mean'].max()

cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-auc-mean']
nround = cv_log.nb.to_dict()[bst_auc]
# thisbst.feature_names is prediction（验证集上真实值和预测值做比较）
preds1 = bst.predict(dtest)
labels = dtest.get_label()
print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))


# In[336]:


#preds1.shape


# In[163]:


clf = xgb.train(params, dtrain, num_boost_round = nround)

# 每个变量对应的重要性
importances = clf.get_fscore()
importances = sorted(importances.items(), key=lambda d: d[1], reverse = True)

# 选择重要性排名前八十的变量
features_chosen = [x[0] for x in importances[:80]]


# In[353]:


features_chosen.append("flag1")
data_thin = dt_s.loc[:, features_chosen]
data_thin.shape


# XGB参数含义及调参方法：https://blog.csdn.net/iyuanshuo/article/details/80142730

# clf = XGBClassifier(
# silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
# #nthread=4,# cpu 线程数 默认最大
# learning_rate= 0.3, # 如同学习率 ，新权值 = 当前权值 - 学习率 × 梯度
# min_child_weight=1
#     #这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#     #假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#     #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
# max_depth=6, # 构建树的深度，越大越容易过拟合
# gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
# subsample=1, # 随机采样训练样本 训练实例的子采样比
# max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
# colsample_bytree=1, # 生成树时进行的列采样 
# reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
# #reg_alpha=0, # L1 正则项参数
# #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
# #objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
# #num_class=10, # 类别数，多分类与 multisoftmax 并用
# n_estimators=100, #树的个数
# seed=1000 #随机种子
# #eval_metric= 'auc')

# In[300]:


#模型保存
#bst.save_model('0001.model')

##计算训练和测试的ks
train_pred = bst.predict(dtrain)
train_labels = dtrain.get_label()
test_pred = bst.predict(dtest)
train_labels = dtest.get_label()

train_perf = sc.perf_eva(y_train, train_pred, title = "train")
test_perf = sc.perf_eva(y_test, test_pred, title = "test")

# PSI
# 建立机器学习模型
#train_p = train_pred.copy()
#test_p  = test_pred.copy()
print("各样本prediction为", preds)


# In[285]:


train_pred.shape


# #### LGB模型
# 

# In[72]:


lgbm_model = LGBMRegressor(
learning_rate =0.05,
n_estimators=500,
max_depth=4,
min_child_weight=0.001,
min_child_samples=20,
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
print(valid_feature_num)


# In[74]:


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''  LGB调优feature_num          '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''feature_num的取值范围根据实际valid_feature_num取值来定'''
feature_num_range=[12,20,50,70,90,100,120,140,160,valid_feature_num]

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
    preds=lgbm_model.predict(P_test.loc[:,chosen_feature]) 
    
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


# In[75]:


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''LGB调优max_depth,min_child_samples, min_split_gain, 确定LGB整体架构  '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

param_test1 = {
'max_depth':list(range(1,6)),#一般情况下max_depth的值为n，2^n 约等于 num_leaves
'min_child_samples':[1,3,5,10,15],
'min_split_gain':[0,0.05,0.1,0.2,0.3,0.4,0.5,0.7,1,2],
}

gsearch1 = GridSearchCV(lgbm_model,param_grid=param_test1,scoring='roc_auc',cv=5)
#gsearch1 = GridsearchCV(lgbm_model,param_grid=param_test1,scoring='roc_auc',cv=5)
starttime = datetime.datetime.now() 
gsearch1.fit(X.loc[:,chosen_final_feature],Y)
endtime = datetime.datetime.now()
print('第一次gridsearch耗时{0} seconds'.format((endtime - starttime).seconds))

gsearch1.grid_scores_,gsearch1.best_params_,gsearch1.best_score_


'''可以直接读取最大auc对应的参数设置，也可以根据观察和经验选取效果好且稳定的参数组合'''
lgbm_model.set_params(max_depth=gsearch1.best_params_['max_depth'])
lgbm_model.set_params(min_child_samples=gsearch1.best_params_['min_child_samples'])
lgbm_model.set_params(min_split_gain=gsearch1.best_params_['min_split_gain'])

'''如果是专家采用经验值调参则不执行上面三行代码，执行下面并给定参数'''
#lgbm_model.set_params(max_depth=5)
#lgbm_model.set_params(min_child_samples=1)
#lgbm_model.set_params(min_split_gain=0.3)


'''选好参数后重新矫正best_n_estimators'''
lgbm_model.set_params(n_estimators=500)
lgbm_param_temp = lgbm_model.get_params()
lgbm_train = lgb.Dataset(X.loc[:,chosen_final_feature],Y)
cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=lgbm_param_temp['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
best_n_estimators=len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)


# In[77]:


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''LGB调优subsample , colsample_bytree    '''''''''''

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
lgbm_model.set_params(subsample=gsearch2.best_params_['subsample'])
lgbm_model.set_params(colsample_bytree=gsearch2.best_params_['colsample_bytree'])

#lgbm_model.set_params(subsample=0.8)
#lgbm_model.set_params(colsample_bytree=0.8)


'''选好参数后重新矫正best_n_estimators'''
lgbm_model.set_params(n_estimators=500)
lgbm_param_temp = lgbm_model.get_params()
lgbm_train = lgb.Dataset(X.loc[:,chosen_final_feature],Y)
cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=lgbm_param_temp['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=100)
best_n_estimators=len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)


# In[78]:


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''LGB调优正则项参数reg_alpha , reg_lambda    '''''''''''

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


# In[79]:


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''LGB调优学习率learning_rate(即XGB模型eta)    '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

param_test4 = {
'learning_rate':[0.001,0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3],
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


print('LGB模型调参完成！最终参数：')
print(lgbm_model.get_params())


# In[296]:


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''' 生成light-gbm model         '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

lgbm_model.fit(X.loc[:,chosen_final_feature],Y,eval_metric='auc')

preds2 = lgbm_model.predict(P_test.loc[:,chosen_final_feature]) 

ks_value,bad_percent,good_percent=pf.cal_ks(-preds2,y_test,section_num=20)                    
max_ks0=np.max(ks_value)
                
false_positive_rate, recall, thresholds = roc_curve(y_test, preds2)
roc_auc0=auc(false_positive_rate,recall) 

print('当前模型在样本内测试集的KS值和AUC值分别为{0}'.format([max_ks0,roc_auc0]))

plt.figure(figsize=(6,2))#后面的图大小有设定，故需提前设定不采用默认值
plt.hist(preds2)
plt.ylabel('Number of samples')
plt.xlabel('probability of y=1')
plt.title('Probability Distribution on test samples')

plt.figure(figsize=(6,2))
plt.plot(list(range(0,21)),np.append([0],bad_percent),'-r',label='Bad Percent')
plt.plot(list(range(0,21)),np.append([0],good_percent),'-g',label='Good Percent')
plt.plot(list(range(0,21)),np.append([0],ks_value),'-b',label='KS value')
plt.legend(loc='lower right')
plt.ylabel('% of total Good/Bad')
plt.xlabel('% of population')

plt.figure(figsize=(6,2))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc0)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()
print("各样本prediction为", preds2)


# In[297]:


lgbm_model.fit(P_test.loc[:,chosen_final_feature],y_test,eval_metric='auc')

test_preds2 = lgbm_model.predict(X.loc[:,chosen_final_feature]) 

ks_value,bad_percent,good_percent=pf.cal_ks(-test_preds2,Y,section_num=20)                    
max_ks0=np.max(ks_value)
                
false_positive_rate, recall, thresholds = roc_curve(Y, test_preds2)
roc_auc0=auc(false_positive_rate,recall) 

print('当前模型在样本内训练集的KS值和AUC值分别为{0}'.format([max_ks0,roc_auc0]))

plt.figure(figsize=(6,2))#后面的图大小有设定，故需提前设定不采用默认值
plt.hist(test_preds2)
plt.ylabel('Number of samples')
plt.xlabel('probability of y=1')
plt.title('Probability Distribution on test samples')

plt.figure(figsize=(6,2))
plt.plot(list(range(0,21)),np.append([0],bad_percent),'-r',label='Bad Percent')
plt.plot(list(range(0,21)),np.append([0],good_percent),'-g',label='Good Percent')
plt.plot(list(range(0,21)),np.append([0],ks_value),'-b',label='KS value')
plt.legend(loc='lower right')
plt.ylabel('% of total Good/Bad')
plt.xlabel('% of population')

plt.figure(figsize=(6,2))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc0)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()
print("各样本prediction为", preds2)


# In[267]:


preds2.shape
train_pred2 = lgbm_model.predict(X.loc[:,chosen_final_feature])


# #### CATBoost模型
# 

# CATboost 官方介绍使用方法以及相关特性：https://tech.yandex.com/catboost/doc/dg/concepts/python-quickstart-docpage/                       
# 主要特征：                                                                                                                               
# 1.需要调整的参数量少                                                                                                                     
# 2.可以直接处理类别型变量，但是如果类别型变量为三要素等唯一型变量仍需删除                                                                 
# 3.独热编码速度比GOSS更快（实际应用速度对比有待商榷）独热编码解释（“噱头”）：https://blog.csdn.net/counsellor/article/details/60145426  
# 4.注意，如果某一列数据中包含字符串值，CatBoost算法就会抛出错误。另外，带有默认值的int型变量也会默认被当成数值数据处理。在CatBoost中，必须对变量进行声明，才可以让算法将其作为分类变量处理。

# In[112]:


one_hot = pd.get_dummies(X)
one_hot = (one_hot - one_hot.mean()) / (one_hot.max() - one_hot.min())
categorical_features_indices = np.where(one_hot.dtypes != np.float)[0]


# In[364]:


model = CatBoostClassifier(
    custom_loss = ['Accuracy'],
    random_seed = 123,
    loss_function = 'MultiClass',
    use_best_model= ['True']#非常重要，iteration数目比较大的时候时间会非常长，所以需要在最优迭代时停下来
)


# In[126]:


model.fit(
    X,Y,
    cat_features = categorical_features_indices,
    verbose = True,  # you can uncomment this for text output
    #plot = True
)


# In[127]:


cat_features = categorical_features_indices


# In[129]:


categorical_features_indices


# In[131]:


feature_score = pd.DataFrame(list(zip(one_hot.dtypes.index, 
                model.get_feature_importance(Pool(one_hot, label=Y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')


# In[152]:


plt.rcParams["figure.figsize"] = (502,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='r')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 6)
ax.set_xlabel('')
rects = ax.patches
labels = feature_score['Score'].round(2)
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')
plt.show()


# In[ ]:


model.score(P_test, y_test)


# Catboost模型调参
# 

# In[372]:


model = CatBoostClassifier(
    l2_leaf_reg = 3,
    iterations = 1000,
    fold_len_multiplier = 1.05,
    learning_rate = 0.03,
    custom_loss = ['Accuracy'],
    random_seed = 100,
    loss_function = 'MultiClass'

)
model.fit(
X,Y,
    cat_features = categorical_features_indices,
    verbose = True,  # you can uncomment this for text output
    plot = True
)
preds_class = model.predict(P_test)
preds_probs = model.predict_proba(P_test)
print('class = ',preds_class)
print('proba = ',preds_probs)


# In[191]:


preds3 = preds_probs[:,1]
preds3.shape


# In[270]:


train_pred3 = (model.predict_proba(X))[:,1]
train_pred3.shape


# In[356]:


params = {'depth': [2, 4, 7, 10],
'learning_rate' : [0.03, 0.1, 0.15],
'l2_leaf_reg': [1,4,9],
'iterations': [300]
'loss_function': 'logloss'         
}
#如果选取的参数组比较多的话，需要的时间特别久，以此模型下的ks值为例可以明显看出过拟合严重，
#故手动调整参数在实际应用中更为合适
cb = ct.CatBoostClassifier(eval_metric="AUC", random_seed=741)
cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = 3)
cb_model.fit(X, Y)
#clf = ct.CatBoostClassifier(eval_metric="AUC", depth=10, iterations= 300, l2_leaf_reg= 9, learning_rate= 0.15)
#clf.fit(X,Y)


# In[382]:


cat_features_index = 2
clf = ct.CatBoostClassifier(eval_metric="AUC", depth=8, iterations= 300, l2_leaf_reg= 10, learning_rate= 0.008)#学习率越大越过拟
clf.fit(X,Y)
preds_class = clf.predict(P_test)
preds_probs = clf.predict_proba(P_test)
print('class = ',preds_class)
print('proba = ',preds_probs)


# #### 计算catboost模型ks值 

# In[383]:


train_pred1 = (clf.predict_proba(X))[:,1]
train_labels1 = dtrain.get_label()
test_pred1 = preds_probs[:,1]
train_labels1 = dtest.get_label()

train_perf1 = sc.perf_eva(y_train, train_pred1, title = "train")
test_perf1 = sc.perf_eva(y_test, test_pred1, title = "test")


# In[384]:


preds3 = preds_probs[:,1]
print(preds3.shape)
train_pred3 = (clf.predict_proba(X))[:,1]
print(train_pred3.shape)


# In[142]:


feature_score = pd.DataFrame(list(zip(one_hot.dtypes.index, model.get_feature_importance(Pool(one_hot, label=Y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])


# In[143]:


feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')


# In[151]:


plt.rcParams["figure.figsize"] = (402,5)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 4)
ax.set_xlabel('')
rects = ax.patches
labels = feature_score['Score'].round(2)
for rect, label in zip(rects, labels):
   height = rect.get_height()
   ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')
plt.show()


# In[153]:


cm = pd.DataFrame()
cm['Satisfaction'] = y_test
cm['Predict'] = model.predict(P_test)


# In[331]:


mappingSatisfaction = {0:'Unsatisfied', 1: 'Neutral', 2: 'Satisfied'}
mappingPredict = {0.0:'Unsatisfied', 1.0: 'Neutral', 2.0: 'Satisfied'}
cm = cm.replace({'Satisfaction': mappingSatisfaction, 'Predict': mappingPredict})
pd.crosstab(cm['Satisfaction'], cm['Predict'], margins=True)


# In[332]:


model.score(P_test, y_test)
print("error rate:","",(1-  model.score(P_test, y_test)))


# 对三种模型的test_preds做一次平均计算
# 

# In[385]:


#print(preds1,preds2,preds3)
preds11 = pd.DataFrame(preds1)
preds12 = pd.DataFrame(preds2)
preds13 = pd.DataFrame(preds3)
'''TD = pd.DataFrame({
    'preds1':[preds11], 
    'preds2':[preds12],
    'preds3':[preds13]

})'''
TD = pd.concat([preds11,preds12,preds13], axis = 1)
predsf = TD.mean(1,round(8))
print(TD)
print("三种模型的平均prediction:",predsf)
predsf = predsf.values
predsf


# 对三种模型的train prediction做一次平均计算

# In[386]:


print(train_pred1,train_pred2,train_pred3)
train_preds11 = pd.DataFrame(train_pred1)
train_preds12 = pd.DataFrame(train_pred2)
train_preds13 = pd.DataFrame(train_pred3)
'''TD = pd.DataFrame({
    'preds1':[preds11], 
    'preds2':[preds12],
    'preds3':[preds13]

})'''
TD = pd.concat([train_preds11,train_preds12,train_preds13], axis = 1)
train_predsf = TD.mean(1,round(8))
print("三种模型的平均prediction:",train_predsf)
train_predsf = train_predsf.values
train_predsf


# #### 根据 

# In[387]:


#模型保存
#bst.save_model('0001.model')

##计算训练和测试的ks
train_pred = train_predsf
train_labels = dtrain.get_label()
test_pred = predsf
train_labels = dtest.get_label()

train_perf = sc.perf_eva(y_train, train_pred, title = "train")
test_perf = sc.perf_eva(y_test, test_pred, title = "test")


# #### 逻辑回归
# 

# In[224]:


pd.concat([preds1,preds2,preds3])

