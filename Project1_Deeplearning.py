
# coding: utf-8

# In[12]:


import sqlite3
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
cnx = sqlite3.connect('database.sqlite')

df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.head()
target = df.pop('overall_rating')
df.shape
target.head()
#target
target.isnull().values.sum()
target.describe()
plt.hist(target, 30, range=(33, 94))
y = target.fillna(target.mean())
y.isnull().values.any()
#Data exploration

df.columns
for col in df.columns:
    unique_cat = len(df[col].unique())
    print("{col}--> {unique_cat}.{typ}".format(col=col, unique_cat=unique_cat, typ=df[col].dtype))
dummy_df = pd.get_dummies(df, columns=['preferred_foot', 'attacking_work_rate', 'defensive_work_rate'])
dummy_df.head()
X = dummy_df.drop(['id', 'date'], axis=1)

#Feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#imputing null value of each column with the mean of that column#imputin 
imput = Imputer()
X_train = imput.fit_transform(X_train)
X_test = imput.fit_transform(X_test)
#finding feature_importance for feature selection. from it we'll be able to decide threshold value
model = XGBRegressor()
model.fit(X_train, y_train)
print(model.feature_importances_)
selection = SelectFromModel(model, threshold=0.01, prefit=True)

select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)

select_X_trainselect_X .shape
#Training Different models
#Linear Regression
pipe = make_pipeline(StandardScaler(),LinearRegression()) 

cv = ShuffleSplit(random_state=0)   

param_grid = {'linearregression__n_jobs': [-1]}    

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)
grid.fit(select_X_train, y_train)
grid.best_params_

#decision tree

lin_reglin_reg  ==  picklepickle.dumpsdumps((gridgrid))

pipe = make_pipeline(StandardScaler(),                 
                     DecisionTreeRegressor(criterion='mse', random_state=0))          

cv = ShuffleSplit(n_splits=10, random_state=42)        

param_grid = {'decisiontreeregressor__max_depth': [3, 5, 7, 9, 13]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)

gridgrid.fitfit((select_X_trainselect_X , y_train)
grid.best_params_
Dectree_reg = pickle.dumps(grid)
                
                #Random forest
pipe = make_pipeline(StandardScaler(),
                     RandomForestRegressor(n_estimators=500, random_state=123))

cv = ShuffleSplit(test_size=0.2, random_state=0)

param_grid = {'randomforestregressor__max_features':['sqrt', 'log2', 10],
              'randomforestregressor__max_depth':[9, 11, 13]}                 

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)


grid.fit((select_X_train, y_train)
grid.best_params_
Randfor_reg = pickle.dumps(grid)
         pipe = make_pipeline(StandardScaler(),
                     XGBRegressor(n_estimators= 500, random_state=42))

cv = ShuffleSplit(n_splits=10, random_state=0)

param_grid = {'xgbregressor__max_depth': [5, 7],
              'xgbregressor__learning_rate': [0.1, 0.3]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs= -1)
grid.fit((select_X_trains, y_train)
grid.best_params_
xgbreg = pickle.dumps(grid)
lin_reg = pickle.loads(lin_reg)
Dectree_reg = pickle.loads(Dectree_reg)
Randfor_reg = pickle.loads(Randfor_reg)
xgbreg = pickle.loads(xgbreg)
print("""Linear Regressor accuracy is {lin}
DecisionTree Regressor accuracy is {Dec}
RandomForest regressor accuracy is {ran}
XGBoost regressor accuracy is {xgb}""".format(lin=lin_reg.score(select_X_test, y_test),
                                                       Dec=Dectree_reg.score(select_X_test, y_test),
                                                       ran=Randfor_reg.score(select_X_test, y_test),
                                                       xgb=xgbreg.score(select_X_test, y_test)))

