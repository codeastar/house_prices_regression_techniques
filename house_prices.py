import pandas as pd

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

def fillNAonDF(df):

    for feat in ('MSZoning', 'Utilities','Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'Electrical'):
        df.loc[:, feat] = df.loc[:, feat].fillna(df[feat].mode()[0])
    
    for feat in ('BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'SaleType'):
        df.loc[:, feat] = df.loc[:, feat].fillna(df[feat].mode()[0])
    
    for feat in ('Alley','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df.loc[:, feat] = df.loc[:, feat].fillna("None")  
    
    for feat in ('MasVnrType', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        df.loc[:, feat] = df.loc[:, feat].fillna("None")   
        
    for feat in ('MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars'):
        df.loc[:, feat] = df.loc[:, feat].fillna(0)    
        
    for feat in ('PoolQC','Fence', 'MiscFeature'):
        df.loc[:, feat] = df.loc[:, feat].fillna("None")
        
    df.loc[:, 'LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

fillNAonDF(df_train)
fillNAonDF(df_test)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df_train = df_train.loc[df_train.GrLivArea < 4000] 


price_dist = sns.distplot(df_train["SalePrice"], color="m", label="Skewness : %.2f"%(df_train["SalePrice"].skew()))
price_dist = price_dist.legend(loc="best")
plt.show()


df_train.loc[:,'SalePrice_log'] = df_train["SalePrice"].map(lambda i: np.log1p(i) if i > 0 else 0)
 
price_log_dist = sns.distplot(df_train["SalePrice_log"], color="m", label="Skewness : %.2f"%(df_train["SalePrice_log"].skew()))
price_log_dist = price_log_dist.legend(loc="best")
plt.show()

def trxNumericToCategory(df):    
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    df['MoSold'] = df['MoSold'].apply(str)

trxNumericToCategory(df_train)
trxNumericToCategory(df_test)

def quantifier(df, feature, df2):
    new_order = pd.DataFrame()
    new_order['value'] = df[feature].unique()
    new_order.index = new_order.value
    new_order['price_mean'] = df[[feature, 'SalePrice_log']].groupby(feature).mean()['SalePrice_log']
    new_order = new_order.sort_values('price_mean')
    new_order = new_order['price_mean'].to_dict()
    
    for categorical_value, price_mean in new_order.items():
        df.loc[df[feature] == categorical_value, feature+'_Q'] = price_mean
        df2.loc[df2[feature] == categorical_value, feature+'_Q'] = price_mean

categorical_features = df_train.select_dtypes(include = ["object"])
   
for f in categorical_features:  
    quantifier(df_train, f, df_test)

new_order = pd.DataFrame()
new_order['value'] = df_train['MSSubClass'].unique()
new_order.index = new_order.value
new_order['price_mean'] = df_train[['MSSubClass', 'SalePrice_log']].groupby('MSSubClass').mean()['SalePrice_log']
MSSubClass_150 = new_order[(new_order.value=='120') | (new_order.value=='160')].price_mean.mean()

df_test.loc[:, "MSSubClass_Q"] = df_test.loc[:, "MSSubClass_Q"].fillna(MSSubClass_150)

def dropCategoricalFeatures(df): 
    df_cat =  df.select_dtypes(include = ["object"])  
    df.drop(df_cat.columns, axis=1, inplace=True)

dropCategoricalFeatures(df_train)
dropCategoricalFeatures(df_test)

train_index = df_train.shape[0]
test_index = df_test.shape[0]

Y_learning = df_train['SalePrice_log']
df_test_id = df_test['Id']

df_all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

df_all_data.drop(['SalePrice'], axis=1, inplace=True)
df_all_data.drop(['SalePrice_log'], axis=1, inplace=True)
df_all_data.drop(['Id'], axis=1, inplace=True)

#skew all features 
def skewFeatures(df):
    skewness = df.skew().sort_values(ascending=False)
    df_skewness = pd.DataFrame({'Skew' :skewness})
    df_skewness = df_skewness[abs(df_skewness) > 0.75]
    df_skewness = df_skewness.dropna(axis=0, how='any')
    skewed_features = df_skewness.index

    for feat in skewed_features:
      df[feat] = np.log1p(df[feat])

skewFeatures(df_all_data)

#find the correlation of features and sale price log
def correlateSalePriceLog(df):
    features = df.columns
    df_corr = pd.DataFrame()
    df_corr['feature'] = features
    df_corr['corre'] = [df[f].corr(df['SalePrice_log']) for f in features]
    df_corr = df_corr.sort_values('corre')
    plt.figure(figsize=(10, 0.3*len(features)))    #w,h in inches
    sns.barplot(data=df_corr, y='feature', x='corre', orient='h')
    plt.show()

correlateSalePriceLog(df_skewed_train)

X_learning = df_all_data[:train_index]
X_test = df_all_data[train_index:]

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LarsCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.linear_model import LassoLars, LassoLarsCV, Ridge, RidgeCV

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

import xgboost as xgb

models = []
models.append(("LrE", LinearRegression() ))
models.append(("RidCV", RidgeCV() ))
models.append(("LarCV", LarsCV() ))
models.append(("LasCV", LassoCV() ))
models.append(("ElNCV", ElasticNetCV() ))
models.append(("LaLaCV", LassoLarsCV() ))
models.append(("XGB", xgb.XGBRegressor() ))

kfold = KFold(n_splits=10)


def getCVResult(models, X_learning, Y_learning):

  for name, model in models:
     cv_results = cross_val_score(model, X_learning, Y_learning, scoring='neg_mean_squared_error', cv=kfold )
     rmsd_scores = np.sqrt(-cv_results)
     print("\n[%s] Mean: %.8f Std. Dev.: %8f" %(name, rmsd_scores.mean(), rmsd_scores.std()))
     k_names.append(name)
     k_means.append(rmsd_scores.mean())
     k_stds.append(rmsd_scores.std())

k_names=[]
k_means=[]
k_stds=[]

getCVResult(models, X_learning, Y_learning)

kfc_df = pd.DataFrame({"RMSD":k_means,"CrossValerrors": k_stds,"Model":k_names})  
sns.barplot("RMSD","Model",data = kfc_df, orient = "h",**{'xerr':k_stds})
plt.show()

#Ridge model tuning
alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5]
models_R = []

for alpha in alphas:
   models_R.append(("Rid_"+str(alpha), Ridge(alpha=alpha) ))

rmsds = getCVResult(models_R, X_learning, Y_learning)

df_ridge = pd.DataFrame(alphas, columns=['alpha'])
df_ridge['rmsd'] = rmsds
sns.pointplot(x="alpha", y="rmsd", data=df_ridge)
plt.show()

#Lasso model tuning 
alphas = [0.000001, 0.000005,0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
models_las = []

for alpha in alphas:
   models_las.append(("Las_"+str(alpha), Lasso(alpha=alpha) ))

rmsds = getCVResult(models_las, X_learning, Y_learning)

df_lasso = pd.DataFrame(alphas, columns=['alpha'])
df_lasso['rmsd'] = rmsds
sns.pointplot(x="alpha", y="rmsd", data=df_lasso)
plt.show()

#Elastic model tuning
alphas = [0.000009, 0.00001, 0.00002, 0.00005, 0.0001, 0.0005, 0.001, 0.005,0.01, 0.05]
models_eln = []

for alpha in alphas:
   models_eln.append(("ELN_"+str(alpha), ElasticNet(alpha=alpha) ))

getCVResult(models_eln, X_learning, Y_learning)

models_eln_l1 = []
l1_ratios = [0.001, 0.005, 0.01,  0.05, 0.1, 0.5, 0.7,0.8, 0.9, 0.99]
for l1_ratio in l1_ratios:
 models_eln_l1.append(("ELN_L1_"+str(l1_ratio), ElasticNet(l1_ratio=l1_ratio, alpha=0.00001) ))

getCVResult(models_eln_l1, X_learning, Y_learning)

#LassoLars tuning 
alphas = [0.000005, 0.00001, 0.00003,0.000035,  0.000036, 0.000037,  0.000038, 0.00004, 0.00005, 0.00007, 0.0001]
models_lala = []

for alpha in alphas:
   models_lala.append(("LaLa_"+str(alpha), LassoLars(alpha=alpha) ))

getCVResult(models_lala, X_learning2, Y_learning)

#XGB model tuning 
n_estimators = [400, 450, 470, 540,550, 560]
models_xgb = []

for n_estimator in n_estimators:
   models_xgb.append(("XGB_"+str(n_estimator), xgb.XGBRegressor(n_estimators=n_estimator,max_depth=3, min_child_weight=3 ) ))

getCVResult(models_xgb, X_learning, Y_learning)

param_test = {
 'max_depth':[3,4,5,7],
 'min_child_weight':[2,3,4]
}
gsearch = GridSearchCV(estimator = xgb.XGBRegressor(n_estimators=470), 
 param_grid = param_test, scoring='neg_mean_squared_error', cv=kfold)
gsearch.fit(X_learning2,Y_learning)
print(gsearch.best_params_ )
print(np.sqrt(-gsearch.best_score_ ))


gammas = [0.0002, 0.0003, 0.00035, 0.0004, 0.0005]
models_xgb_gamma = []

for gamma in gammas:
   models_xgb_gamma.append(("XGB_"+str(gamma), xgb.XGBRegressor(n_estimators=470,max_depth=3, min_child_weight=3, gamma=gamma,subsample=0.5 ) ))

getCVResult(models_xgb_gamma, X_learning, Y_learning)

param_test = {
 'subsample':[0.1, 0.15 ,0.45,0.5, 0.55],
 'colsample_bytree':[0.75, 0.9, 0.95,1]
}
gsearch = GridSearchCV(estimator = xgb.XGBRegressor(n_estimators=470,max_depth=3, min_child_weight=3, gamma=0.0003), 
 param_grid = param_test, scoring='neg_mean_squared_error', cv=kfold)
gsearch.fit(X_learning2,Y_learning)
print(gsearch.best_params_ )
print(np.sqrt(-gsearch.best_score_ ))

param_test = {
 'reg_alpha':[0.45, 0.5, 0.53, 0.6],
 'reg_lambda':[0.75, 0.8, 0.85]
}
gsearch = GridSearchCV(estimator = xgb.XGBRegressor(n_estimators=470,max_depth=3, min_child_weight=3, gamma=0.0003, subsample=0.5), 
 param_grid = param_test, scoring='neg_mean_squared_error', cv=kfold)
gsearch.fit(X_learning2,Y_learning)
print(gsearch.best_params_ )
print(np.sqrt(-gsearch.best_score_ ))

learns = [0.04, 0.041, 0.042 ,0.043, 0.048]
models_xgb_learn = []

for learn in learns:
   models_xgb_learn.append(("XGB_"+str(learn), xgb.XGBRegressor(n_estimators=470,max_depth=3, min_child_weight=3, 
                                                                learning_rate=learn,subsample=0.5, 
                                                               reg_alpha=0.5,reg_lambda=0.8) ))

getCVResult(models_xgb_learn, X_learning, Y_learning)


#get results from tuned model

tuned_models = []
tuned_models.append(("Rid_t", Ridge(alpha=0.01) ))
tuned_models.append(("Las_t", Lasso(alpha=0.00001) ))
tuned_models.append(("ElN_t", ElasticNet(l1_ratio=0.8, alpha=0.00001) ))
tuned_models.append(("LaLa_t", LassoLars(alpha=0.000037) ))
tuned_models.append(("XGB_t", xgb.XGBRegressor(n_estimators=470,max_depth=3, min_child_weight=3, 
                                                                learning_rate=0.042,subsample=0.5, 
                                                               reg_alpha=0.5,reg_lambda=0.8)  ))


getCVResult(tuned_models, X_learning, Y_learning)

#ensemble model with stacking
linearM = LinearRegression()
ridM = Ridge(alpha=0.01)
lasM = Lasso(alpha=0.00001)
elnM = ElasticNet(l1_ratio=0.8, alpha=0.00001)
lalaM = LassoLars(alpha=0.000037)
xgbM =  xgb.XGBRegressor(n_estimators=470,max_depth=3, min_child_weight=3, 
                                                                learning_rate=0.042,subsample=0.5, 
                                                               reg_alpha=0.5,reg_lambda=0.8) 

stack_kfold = KFold(n_splits=10, shuffle=True) 

base_models = []
base_models.append(lasM)
base_models.append(ridM)
base_models.append(xgbM)
base_models.append(elnM)
base_models.append(linearM)

meta_model = lalaM

#fill up all zero
kf_predictions = np.zeros((X_learning.shape[0], len(base_models)))

#get the value part
X_values = X_learning.values
Y_values = Y_learning.values

for i, model in enumerate(base_models):
    for train_index ,test_index in stack_kfold.split(X_values):
        model.fit(X_values[train_index], Y_values[train_index])
        model_pred = model.predict(X_values[test_index])
        kf_predictions[test_index, i] = model_pred   

#teach the meta model        
meta_model.fit(kf_predictions, Y_values)        
  
preds = []

for model in base_models:
    model.fit(X_learning, Y_learning)
    pred = model.predict(X_test)
    preds.append(pred)

base_predictions = np.column_stack(preds)

#get stacked prediction
stacked_predict = meta_model.predict(base_predictions)

def getPred(model, X, Y, test):
    model.fit(X, Y)
    pred = model.predict(test)
    return pred

def getCSV(file_name, id_col, pred):
    sub = pd.DataFrame()
    sub['Id'] = id_col
    sub['SalePrice'] = np.expm1(pred)
    sub.to_csv(file_name,index=False)

#get other tuned models prediction    
xgb_pred =  getPred(xgbM, X_learning, Y_learning, X_test)    
eln_pred =  getPred(elnM, X_learning, Y_learning, X_test)    
rid_pred =  getPred(ridM, X_learning, Y_learning, X_test)    

#percentage is based on each model's CV scores
stack_n_tuned_prediction = stacked_predict *0.5 + xgb_pred * 0.3 + eln_pred *0.1+ rid_pred *0.1

getCSV('stack_n_tuned_prediction.csv',df_test['Id'], stack_n_tuned_prediction )

#got the CSV then submit to Kaggle
