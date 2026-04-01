# ---- Cell 1 ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from pyod.models.knn import KNN
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# ---- Cell 2 ----
#pip install pyod

# ---- Cell 3 ----
data_arc = pd.read_csv(r'C:\Users\MaYu7006\Documents\info_math\yandex ds\final\data_arc.csv')
data_arc.head(20)

# ---- Cell 4 ----
data_arc.info()

# ---- Cell 5 ----
data_arc['key'].unique()

# ---- Cell 6 ----
data_arc.isnull().sum()

# ---- Cell 7 ----
data_arc['Активная мощность'].describe()

# ---- Cell 8 ----
data_arc['Активная мощность'].hist(bins=100, figsize=(10, 8))

# ---- Cell 9 ----
data_arc['Реактивная мощность'].describe()

# ---- Cell 10 ----
data_arc[data_arc['Реактивная мощность'] < 0].count()

# ---- Cell 11 ----
data_arc = data_arc[data_arc["Реактивная мощность"] > 0]

# ---- Cell 12 ----
data_arc['Реактивная мощность'].describe()

# ---- Cell 13 ----
data_arc['Реактивная мощность'].hist(bins=100, figsize=(10, 8))

# ---- Cell 14 ----
data_bulk = pd.read_csv(r'C:\Users\MaYu7006\Documents\info_math\yandex ds\final\data_bulk.csv')
data_bulk.head(10)

# ---- Cell 15 ----
data_bulk.info()

# ---- Cell 16 ----
for column in data_bulk.columns:
    print(column)
    display(data_bulk[column].describe())

# ---- Cell 17 ----
data_bulk[column].hist(bins=100, figsize=(10, 8), alpha=0.6)

# ---- Cell 18 ----
for column in data_bulk.columns:
    if column[0] == 'B':
        data_bulk[column].hist(bins=100, figsize=(10, 8), alpha=0.6)

# ---- Cell 19 ----
data_bulk_time = pd.read_csv(r'C:\Users\MaYu7006\Documents\info_math\yandex ds\final\data_bulk_time.csv')
data_bulk_time.head(10)

# ---- Cell 20 ----
data_bulk_time.info()

# ---- Cell 21 ----
for column in data_bulk_time.columns:
    print(column)
    display(data_bulk_time[column].describe())

# ---- Cell 22 ----
data_bulk_time['key'].unique()

# ---- Cell 23 ----
data_gas = pd.read_csv(r'C:\Users\MaYu7006\Documents\info_math\yandex ds\final\data_gas.csv')
data_gas.head(10)

# ---- Cell 24 ----
data_gas.describe()

# ---- Cell 25 ----
data_gas['key'].unique()

# ---- Cell 26 ----
data_gas.isnull().sum()

# ---- Cell 27 ----
data_gas['Газ 1'].hist(bins=100, figsize=(10, 8))

# ---- Cell 28 ----
data_temp = pd.read_csv(r'C:\Users\MaYu7006\Documents\info_math\yandex ds\final\data_temp.csv')
data_temp.head(10)

# ---- Cell 29 ----
data_temp.info()

# ---- Cell 30 ----
data_temp['key'].unique()

# ---- Cell 31 ----
data_temp['Температура'].describe()

# ---- Cell 32 ----
data_temp.isnull().sum()

# ---- Cell 33 ----
data_temp['Температура'].hist(bins=100, figsize=(10,8))

# ---- Cell 34 ----
data_wire = pd.read_csv(r'C:\Users\MaYu7006\Documents\info_math\yandex ds\final\data_wire.csv')
data_wire.head(10)

# ---- Cell 35 ----
data_wire.info()

# ---- Cell 36 ----
data_wire['key'].unique()

# ---- Cell 37 ----
for column in data_wire.columns:
    print(column)
    display(data_wire[column].describe())

# ---- Cell 38 ----
for column in data_wire.columns:
    if column[0] == 'W':
        data_wire[column].hist(bins=100, figsize=(16,10), alpha=0.3)

# ---- Cell 39 ----
data_wire_time = pd.read_csv(r'C:\Users\MaYu7006\Documents\info_math\yandex ds\final\data_wire_time.csv')
data_wire_time.head(10)

# ---- Cell 40 ----
data_wire_time.info()

# ---- Cell 41 ----
data_wire_time['key'].unique()

# ---- Cell 42 ----
for column in data_wire_time.columns:
    print(column)
    display(data_wire_time[column].describe())

# ---- Cell 43 ----
data_arc.columns = ['key', 'start_time','end_time', 'active_power', 'reactive_power']
data_arc_all = pd.pivot_table(data_arc,
                             values=['active_power','reactive_power'],
                             index='key',
                             aggfunc={'active_power': np.sum,
                                      'reactive_power': np.sum})
data_arc_all.columns = ['sum_active_power','sum_reactive_power']
data_arc_all.head()

# ---- Cell 44 ----
bad = []
for key in list(data_temp['key'].unique()):
    try:
        if ((data_temp[data_temp['key'] == key]['Время замера'].max() < 
            data_arc[data_arc['key'] == key]['end_time'].max()) or
           (data_temp[data_temp['key'] == key]['Время замера'].max() == 
            data_temp[data_temp['key'] == key]['Время замера'].min())):
            bad.append(key)
    except:
        bad.append(key)

# ---- Cell 45 ----
data_bulk = data_bulk.set_index('key')

# ---- Cell 46 ----
data_bulk.columns = ['bulk_1', 'bulk_2', 'bulk_3', 'bulk_4', 'bulk_5', 'bulk_6', 'bulk_7', 'bulk_8', 'bulk_9', 'bulk_10', 'bulk_11', 'bulk_12', 'bulk_13', 'bulk_14', 'bulk_15']

# ---- Cell 47 ----
data_bulk = data_bulk.drop(columns= 'bulk_8')

# ---- Cell 48 ----
data_gas = data_gas.set_index('key')
data_gas.columns = ['gas']

# ---- Cell 49 ----
data_wire = data_wire.set_index('key')
data_wire.columns = ['wire_1', 'wire_2', 'wire_3', 'wire_4', 'wire_5', 'wire_6', 'wire_7', 'wire_8', 'wire_9']
data_wire = data_wire.drop(columns= 'wire_5')

# ---- Cell 50 ----
data_temp = data_temp.dropna()
data_temp.info()

# ---- Cell 51 ----
data_temp = data_temp.query('key not in @bad')
data_temp.info()

# ---- Cell 52 ----
count = (data_temp['key'].value_counts() < 2).sum() 
count

# ---- Cell 53 ----
bad_key = list(data_temp['key'].value_counts().index[:count])
bad_key

# ---- Cell 54 ----
data_temp = data_temp.query('key not in @bad_key')

# ---- Cell 55 ----
data_temp.head()

# ---- Cell 56 ----
data_temp_time = pd.pivot_table(data_temp,
                                values='Время замера',
                                index='key',
                                aggfunc={'Время замера': [np.min, np.max]})
data_temp_time

# ---- Cell 57 ----
data_temp.columns = ['key', 'time','temp']

# ---- Cell 58 ----
end= list(data_temp_time['amax'])
start = list(data_temp_time['amin'])
temp_end = data_temp.query('time in @end')
temp_end = temp_end.set_index('key')
temp_start= data_temp.query('time in @start')
temp_start = temp_start.set_index('key')
data_temp_all = temp_start.copy()
data_temp_all.columns = ['time','start_t']
data_temp_all['end_t'] = temp_end['temp']
data_temp_all = data_temp_all.drop('time', axis=1)
data_temp_all.head()

# ---- Cell 59 ----
data_final = pd.concat([data_arc_all, data_temp_all, data_bulk, data_gas, data_wire], axis=1, sort=False)
# data_final = data_steel.dropna(subset=['end_temp'])
data_final.head()

# ---- Cell 60 ----
data_final.info()

# ---- Cell 61 ----
data_final = data_final.fillna(0)

# ---- Cell 62 ----
datas = data_final.query('start_t == 0 and start_t == 0')
datas

# ---- Cell 63 ----
data_final = data_final.query('start_t != 0 and start_t != 0')
data_final

# ---- Cell 64 ----
data_final = data_final.astype(
    {"start_t":'int16', 
     "end_t":'int16',
     "bulk_1":'int16',
     "bulk_2":'int16',
     "bulk_3":'int16',
     "bulk_4":'int16',
     "bulk_5":'int16',
     "bulk_6":'int16',
     "bulk_7":'int16',
     "bulk_9":'int16',
     "bulk_10":'int16',
     "bulk_11":'int16',
     "bulk_12":'int16',
     "bulk_13":'int16',
     "bulk_14":'int16',
     "bulk_15":'int16'})

# ---- Cell 65 ----
data_final.info()

# ---- Cell 66 ----
corrmat = data_final.corr()

  

f, ax = plt.subplots(figsize =(9, 8))

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

# ---- Cell 67 ----
data_final.corr()['end_t']

# ---- Cell 68 ----
data_model = data_final.copy()
data_model

# ---- Cell 69 ----
data_final

# ---- Cell 70 ----
data_model = data_model.drop('sum_reactive_power', axis=1)
data_model = data_model.drop('bulk_1', axis=1)
data_model = data_model.drop('bulk_4', axis=1)
data_model = data_model.drop('bulk_5', axis=1)
data_model = data_model.drop('bulk_7', axis=1)
data_model = data_model.drop('bulk_9', axis=1)
data_model = data_model.drop('bulk_10', axis=1)
data_model = data_model.drop('bulk_13', axis=1)
data_model = data_model.drop('bulk_14', axis=1)
data_model = data_model.drop('bulk_15', axis=1)
data_model = data_model.drop('gas', axis=1)
data_model = data_model.drop('wire_1', axis=1)
data_model = data_model.drop('wire_3', axis=1)
data_model = data_model.drop('wire_4', axis=1)
data_model = data_model.drop('wire_6', axis=1)
data_model = data_model.drop('wire_8', axis=1)
data_model = data_model.drop('wire_9', axis=1)
data_model.head()

# ---- Cell 71 ----
data_model

# ---- Cell 72 ----
from sklearn import tree
import seaborn as sns
%matplotlib inline

# ---- Cell 73 ----
max_depth_values=range(1,100)
scores_data = pd.DataFrame()
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)

# ---- Cell 74 ----
model

# ---- Cell 75 ----
data_model.info()

# ---- Cell 76 ----
random_state = 4102020
# features = data_steel.drop('end_temp', axis=1)
# target = data_steel['end_temp']

features = data_model.drop('end_t', axis=1)
target = data_model['end_t']

features_train, features_test, target_train, target_test = train_test_split(
                                                            features, 
                                                            target, 
                                                            test_size=0.25, 
                                                            random_state=random_state)
cv_counts = 5

# ---- Cell 77 ----
joblib.dump(model, 'slope_from_sentiment_model.pkl')

mae = mean_absolute_error(target_test, model.predict(features_test))
mae

# ---- Cell 78 ----
model = LogisticRegression()
model.fit(features_train, target_train)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

# ---- Cell 79 ----
import joblib

# ---- Cell 80 ----
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

# ---- Cell 81 ----
model.fit(features_train, target_train)

# ---- Cell 82 ----
joblib.dump(model, 'slope_from_sentiment_model.pkl')

mae = mean_absolute_error(target_train, model.predict(features_train))
mae

# ---- Cell 83 ----
print(features_train.shape)
print(features_test.shape)
print(target_train.shape)
print(target_test.shape)

# ---- Cell 84 ----
features_train

# ---- Cell 85 ----
features

# ---- Cell 86 ----
model = LinearRegression()

MAE = (cross_val_score(model, 
                             features_train, 
                             target_train, 
                             cv=cv_counts, 
                             scoring='neg_mean_absolute_error').mean() * -1)
print('MAE model LinearRegression =', MAE)

# ---- Cell 87 ----
model.fit(features_train, target_train)

# ---- Cell 88 ----
joblib.dump(model, 'slope_from_sentiment_model.pkl')

mae = mean_absolute_error(target_train, model.predict(features_train))
mae

# ---- Cell 89 ----
best_model = None
best_result = 10000
best_est = 0
best_depth = 0
for est in range(1, 51, 10):
    for depth in range (1, 20):
        model = RandomForestRegressor(random_state=12345, n_estimators=est, max_depth=depth) 
        model.fit(features_train, target_train) 
        predictions_test= model.predict(features_test) 
        result = mean_absolute_error(target_test, predictions_test)
        if result < best_result:
            best_model = model
            best_result = result
            best_est = est
            best_depth = depth

print("MAE наилучшей модели:", best_result, "Количество деревьев:", best_est, "Максимальная глубина:", depth)

# ---- Cell 90 ----
model.fit(features_train, target_train)

# ---- Cell 91 ----
joblib.dump(model, 'slope_from_sentiment_model.pkl')

mae = mean_absolute_error(target_train, model.predict(features_train))
mae

# ---- Cell 92 ----
model.fit(features_test, target_test)

# ---- Cell 93 ----
joblib.dump(model, 'slope_from_sentiment_model.pkl')

mae = mean_absolute_error(target_test, model.predict(features_test))
mae

# ---- Cell 94 ----
model = CatBoostRegressor(verbose=False, random_state=random_state)

MAE = (cross_val_score(model, 
                             features_train, 
                             target_train, 
                             cv=cv_counts, 
                             scoring='neg_mean_absolute_error').mean() * -1)
print('Mean MAE from CV of CatBoostRegressor =', MAE)

# ---- Cell 95 ----
model.fit(features_train, target_train)

# ---- Cell 96 ----
joblib.dump(model, 'slope_from_sentiment_model.pkl')

mae = mean_absolute_error(target_train, model.predict(features_train))
mae

# ---- Cell 97 ----
#pip install catboost

# ---- Cell 98 ----
from catboost import CatBoostRegressor

# ---- Cell 99 ----
model = LinearRegression()
model.fit(features_train, target_train)
target_predict = model.predict(features_test)
test_MAE_LR = mean_absolute_error(target_predict, target_test)
print('MAE on test for LinearRegression =', test_MAE_LR)

# ---- Cell 100 ----
model.fit(features_train, target_train)

# ---- Cell 101 ----
joblib.dump(model, 'slope_from_sentiment_model.pkl')

mae = mean_absolute_error(target_train, model.predict(features_train))
mae

# ---- Cell 102 ----
model = CatBoostRegressor(verbose=False)
model.fit(features_train, target_train)
target_predict = model.predict(features_test)
test_MAE_CBR = mean_absolute_error(target_predict, target_test)
print('MAE on test of CatBoostRegressor =', test_MAE_CBR)

# ---- Cell 103 ----
model.fit(features_train, target_train)

# ---- Cell 104 ----
joblib.dump(model, 'slope_from_sentiment_model.pkl')

mae = mean_absolute_error(target_train, model.predict(features_train))
mae
