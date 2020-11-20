import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from math import sqrt
import math
from sklearn.metrics import mean_squared_error as MSE
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute #用KNN填補空缺值
train_data = pd.read_csv('D:/專題/House Prices Advanced Regression Techniques/train.csv',engine='python')
test_data = pd.read_csv('D:/專題/House Prices Advanced Regression Techniques/test.csv',engine='python')

Id=test_data['Id']

num_train=len(train_data)
num_test=len(test_data)

train_data.head()
test_data.head()
display(train_data.shape)
display(test_data.shape)
print(train_data.describe())
print(train_data.isnull().sum())  
print(test_data.isnull().sum())

train_data.drop('Id',1,inplace=True)
test_data.drop('Id',1,inplace=True)

# A Barplot with the most correlated with the target
correlations =train_data.corr().abs()['SalePrice'].sort_values(ascending=False)[1:]
ax = sns.barplot(x=correlations.values,y=correlations.index).set_title('Most Correlated with SalePrice')

high_cor_feature=[]
for i in range(len(correlations)):
    if correlations[i]>0.3:
        high_cor_feature.append(correlations.index[i])
        
    

#將目標預測變數分離
y=train_data['SalePrice'].reset_index(drop=True)
train_data=train_data.drop(['SalePrice'],axis=1)
#combine all data to deal with NAs
data=pd.concat([train_data,test_data],ignore_index=True)
#看缺失變數的缺失程度
def missing_percentage(df):
    
    '''A function for showing missing data values'''
    
    total = df.isnull().sum().sort_values(
        ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = (df.isnull().sum().sort_values(ascending=False) / len(df) *
               100)[(df.isnull().sum().sort_values(ascending=False) / len(df) *
                     100) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_feature=missing_percentage(data)

#對變數缺失程度作圖
fig, ax = plt.subplots(figsize=(20, 5))
sns.barplot(x=missing_feature.index, y='Percent', data=missing_feature, palette="vlag")
plt.xticks(rotation=90)# 使x軸文字不重疊，變垂直
#display(missing_feature.T.style.background_gradient(cmap='Blues', axis=1))

#取所有col name 
col_name=data.columns.values.tolist() 
#取missing value 
missing_name=missing_feature.index.tolist()

#刪掉缺失值>60%的變數
for feature in missing_name:
    if(missing_feature['Percent'][feature]>60):
        data.drop(feature,axis=1,inplace=True)

col_name=data.columns.values.tolist() 
missing_feature=missing_percentage(data)
#刪掉缺失值>60%的變數之作圖
fig, ax = plt.subplots(figsize=(20, 5))
sns.barplot(x=missing_feature.index, y='Percent', data=missing_feature, palette='Reds_r')
plt.xticks(rotation=90)
display(missing_feature.T.style.background_gradient(cmap='Reds', axis=1))

missing_name=missing_feature.index.tolist()

#區分個變數的型態
for feature in missing_name:
    print(type(data[feature][1]))
#正式分離各個變數型態
missing_feature_numpy_float64=[]
missing_feature_str=[]
for feature in missing_name:
    if(type(data[feature][1])==np.float64):
        missing_feature_numpy_float64.append(feature)
    elif(type(data[feature][1])==str):
        missing_feature_str.append(feature)

data_2=data.copy()
#使用KNN填補空缺值
for feature in col_name:
    if(type(data_2[feature][0])!=np.float64):
        data_2[feature]=data_2[feature].astype('category').cat.codes
    '''for i in range(len(data_2[feature])):
        if(data_2[feature][i]==-1):
            data_2[feature][i]=None'''
    data_2.loc[data_2[feature]==-1,feature] = None
data_fill_NA_by_KNN=KNN(k=5).fit_transform(data_2)    

#數值型變數用中位數填
for feature in missing_feature_numpy_float64:
    data[feature]=data[feature].fillna(data[feature].median())
#類別型變數用眾數填
#求出眾數        
def mode(series):
    #統計個數值出現次數
    number_dict={}
    for i in range(len(series)):
        if(series[i]==-1):#-1 代表的是Nan，因此不納入統計
            continue
        elif str(series[i]) in number_dict:
                number_dict[str(series[i])]+=1
        else:
            number_dict[str(series[i])]=1
    #求眾數
    max_appear=0
    for times in number_dict.values():
        if times > max_appear:
            max_appear=times
    mode_list=[]
    for key, value in number_dict.items():
        if value==max_appear:
            mode_list.append(key)
    return int(mode_list[0])

#transform_data=data['FireplaceQu'].astype('category').cat.codes #Nan會變-1
#imputer=Imputer(missing_values=-1,strategy='most_frequent')
#data["FireplaceQu"]=imputer.fit_transform(transform_data)
#眾數
#counts = np.bincount(transform_data) #不可有負數
#np.argmax(counts)

#sns.countplot(transform_data) #只能單一慢慢看
#將-1改為眾數    
for feature in missing_feature_str:
    data[feature]=data[feature].astype('category').cat.codes
    mode_num=mode(data[feature])
    #print(mode_num)
    #for i in range(len(data[feature])):#此法會跳錯誤訊息
        #if data[feature][i]==-1:
            #data[feature][i]=mode_num 
    data.loc[data[feature]==-1,feature] = mode_num  
    
#data.loc[data['FireplaceQu']==2,'FireplaceQu']          
 #確認是否填補完 
missing_feature=missing_percentage(data)
print(missing_feature)

for feature in col_name:
    if(type(data[feature][1])==str):
        data[feature]=data[feature].astype('category').cat.codes
        
#挑重要變數的data    
#dataset_high_cor= data[high_cor_feature]      

#變回原本正常的  training set和testing set
training_data=data[0:num_train]
testing_data=data[num_train:]

#training_data_high_cor=dataset_high_cor[0:num_train]
#testing_data_high_cor=dataset_high_cor[num_train:]



#將資料標準化 .StandardScaler() (也可用正規化 .MinMaxScaler()) 用training data的標準化
#https://ithelp.ithome.com.tw/articles/10216967?sc=iThelpR
#scaler.transform(X) 
#scaler.inverse_transform(scaler.transform(X)) 轉回來
#要將全部資料一起標準化，還是要將train test 分開呢

scaler=preprocessing.RobustScaler().fit(training_data)
training_data=pd.DataFrame(scaler.transform(training_data),columns=col_name)

testing_data=pd.DataFrame(scaler.transform(testing_data),columns=col_name)
#挑重要變數的data 
training_data_high_cor=training_data[high_cor_feature] 
testing_data_high_cor=testing_data[high_cor_feature]

#再將training data 切分，做cross validation
x_train, x_test, y_train, y_test = train_test_split(training_data, y, test_size=0.2, random_state=0)
# logistic regression
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(x_train,y_train)
prediction_svr=svr.predict(x_test)

#XG Boost
xgb_r = xgb.XGBRegressor(objective ='reg:linear', n_estimators = 10, seed = 123) 

xgb_r.fit(x_train,y_train)
prediction_xgb=xgb_r.predict(x_test)

#Random Forest Regression
rf_r=RandomForestRegressor(n_estimators = 100 , oob_score = True, random_state = 42)

rf_r.fit(x_train,y_train)
prediction_rf=rf_r.predict(x_test)

#RMSE function
def RMSE(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 
        absError.append(abs(val))#誤差絕對值
    return sqrt(sum(squaredError) / len(squaredError)) #sum(absError) / len(absError))#平均絕對誤差MAE


rmse_svr=RMSE(y_test.values,prediction_svr)
rmse_xgb=RMSE(y_test.values,prediction_xgb)
rmse_rf=RMSE(y_test.values,prediction_rf)
#np.sqrt(MSE(y_test.values,prediction_svr)) #較快算出RMSE的方法


models = pd.DataFrame({
    'Model': ['Support Vector Regression','XGBoost Regression','Random Forest Regression'],
    'RMSE': [rmse_svr,rmse_xgb,rmse_rf]
    })

models.sort_values(by='RMSE', ascending=True)

#XGB for all data feature
'''xgb_r = xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
xgb_r.fit(training_data,y)
prediction_xgb_formal=xgb_r.predict(testing_data)

final_answer=pd.DataFrame({'Id':Id,'SalePrice':prediction_xgb_formal})
final_answer.to_csv('python_xgb_HousePrice.csv',index=False)

#XGB for high cor feature
xgb_r.fit(training_data_high_cor,y)
prediction_xgb_formal=xgb_r.predict(testing_data_high_cor)

final_answer=pd.DataFrame({'Id':Id,'SalePrice':prediction_xgb_formal})
final_answer.to_csv('python_xgb_high_cor_HousePrice.csv',index=False)'''

#Random Forest for all data feature
rf_r=RandomForestRegressor(n_estimators = 100 , oob_score = True, random_state = 42)
rf_r.fit(training_data,y)
prediction_rf_formal=rf_r.predict(testing_data)

final_answer=pd.DataFrame({'Id':Id,'SalePrice':prediction_rf_formal})
final_answer.to_csv('python_rf_HousePrice.csv',index=False)


#SVR for all data feature
'''svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(training_data,y)
prediction_svr_formal=svr.predict(testing_data)       
        
final_answer=pd.DataFrame({'Id':Id,'SalePrice':prediction_svr_formal})
final_answer.to_csv('python_svr_HousePrice.csv',index=False)'''
        
# SVR for high cor feature
'''svr = SVR(kernel='rbf', C=1e3, gamma=1e-8)
svr.fit(training_data_high_cor,y)
prediction_svr_high_cor_formal=svr.predict(testing_data_high_cor)    
final_answer=pd.DataFrame({'Id':Id,'SalePrice':prediction_svr_high_cor_formal})
final_answer.to_csv('python_svr_HighCor_HousePrice.csv',index=False)'''      
        
###########################################
#data filling NA by KNN
training_data=data_2[0:num_train]
testing_data=data_2[num_train:]

xgb_r = xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
xgb_r.fit(training_data,y)
prediction_xgb_formal=xgb_r.predict(testing_data)

final_answer=pd.DataFrame({'Id':Id,'SalePrice':prediction_xgb_formal})
final_answer.to_csv('python_xgb_filling_NA_KNN_HousePrice.csv',index=False)

      
