import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd
from sklearn.ensemble import GradientBoostingRegressor


fraction = 1.
''' Grad Boost simple '''
'''
#Remove, ne doit pas intervenir dans la prediction
train_input.drop(['station_id', 'ID'], axis=1, inplace=True)
test_input.drop(['station_id', 'ID'], axis=1, inplace=True)
grad_boost = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500, subsample=1.0, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=10, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

# sample aléatoire pour la cross validation
rindex = np.array(rd.sample(train_input.index, np.int(fraction*len(train_input))))
X = train_input.iloc[rindex,:].copy()
Y = train_output.loc[rindex,"TARGET"]
X_test = train_input.drop(rindex).copy()
Y_test = train_output.drop(rindex).TARGET

X.drop(['station_id', 'ID'], axis=1, inplace=True)
X_test.drop(['station_id', 'ID'], axis=1, inplace=True)

# Lin reg

grad_boost.fit(X,Y)

print('MSE train = ' +str(score_function(Y, grad_boost.predict(X))))
print('MSE test = ' +str(score_function(Y_test, grad_boost.predict(X_test))))
# meilleur 35.8190438937 learning_rate=0.1, n_estimators=500 max_depth=10,

# sur le test
res = pd.DataFrame(grad_boost.predict(test_input), columns = ['TARGET'])
res['ID'] = res.index
res = res[['ID', 'TARGET']]
res.to_csv("F:\MVA\SW\Outputs\challenge_output_V5.csv" ,sep = ',', index=False)

#save model
import pickle
pickle.dump(grad_boost, open("F:\MVA\SW\Models\GB\GB_Global_V2.dat", "wb"))

'''

''' Triple Grad Boost '''
# Selection train-test par rows

rindex = np.array(rd.sample(train_input_NO2.index, np.int(0.7*len(train_input_NO2))))
X = train_input_NO2.iloc[rindex,:].copy()
Y = train_output_NO2.iloc[rindex].TARGET
X_test = train_input_NO2.drop(rindex).copy()
Y_test = train_output_NO2.drop(rindex).TARGET
'''
# Selection train-test par station
stations_out = np.array(rd.sample(train_input_NO2.station_id.unique(), 4))
sindex = np.where((train_input_NO2.station_id == stations_out[0]) | (train_input_NO2.station_id == stations_out[1]) | (train_input_NO2.station_id == stations_out[2]) | (train_input_NO2.station_id == stations_out[3]))[0]

X = train_input_NO2.drop(sindex).copy()
Y = train_output_NO2.drop(sindex).TARGET
X_test = train_input_NO2.iloc[sindex,:].copy()
Y_test = train_output_NO2.iloc[sindex].TARGET
'''
X.drop(['station_id', 'ID'], axis=1, inplace=True)
X_test.drop(['station_id', 'ID'], axis=1, inplace=True)
# grad boost
grad_boost_NO2 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=700, subsample=0.6, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=10, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=0.8, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

grad_boost_NO2.fit(X,Y)
square_error_NO2_train = (grad_boost_NO2.predict(X)-Y)**2
square_error_NO2 = (grad_boost_NO2.predict(X_test)-Y_test)**2
print("MSE train NO2 = "+str(np.mean(square_error_NO2_train)))
print("MSE test NO2 = "+str(np.mean(square_error_NO2)))
# meilleur 35.8153639899 learning_rate=0.1, n_estimators=500 max_depth=10,

# Selection train-test par rows
rindex = np.array(rd.sample(train_input_PM10.index, np.int(0.7*len(train_input_PM10))))
X = train_input_PM10.iloc[rindex,:].copy()
Y = train_output_PM10.iloc[rindex].TARGET
X_test = train_input_PM10.drop(rindex).copy()
Y_test = train_output_PM10.drop(rindex).TARGET
'''

# Selection train-test par station
stations_out = np.array(rd.sample(train_input_PM10.station_id.unique()[train_input_PM10.station_id.unique() != 22], 4))
sindex = np.where((train_input_PM10.station_id == stations_out[0]) | (train_input_PM10.station_id == stations_out[1]) | (train_input_PM10.station_id == stations_out[2]) | (train_input_PM10.station_id == stations_out[3]))[0]

X = train_input_PM10.drop(sindex).copy()
Y = train_output_PM10.drop(sindex).TARGET
X_test = train_input_PM10.iloc[sindex,:].copy()
Y_test = train_output_PM10.iloc[sindex].TARGET
'''
X.drop(['station_id', 'ID'], axis=1, inplace=True)
X_test.drop(['station_id', 'ID'], axis=1, inplace=True)
# grad boost
grad_boost_PM10 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=700, subsample=0.6, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=10, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=0.8, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
grad_boost_PM10.fit(X,Y)
square_error_PM10_train = (grad_boost_PM10.predict(X)-Y)**2
square_error_PM10 = (grad_boost_PM10.predict(X_test)-Y_test)**2
print("MSE train PM10 = "+str(np.mean(square_error_PM10_train)))
print("MSE test PM10 = "+str(np.mean(square_error_PM10)))
# meilleur 13.3930515372 learning_rate=0.2, n_estimators=500 max_depth=10,


# Selection train-test par rows
rindex = np.array(rd.sample(train_input_PM2_5.index, np.int(0.7*len(train_input_PM2_5))))
X = train_input_PM2_5.iloc[rindex,:].copy()
Y = train_output_PM2_5.iloc[rindex].TARGET
X_test = train_input_PM2_5.drop(rindex).copy()
Y_test = train_output_PM2_5.drop(rindex).TARGET
'''
# Selection train-test par station
stations_out = np.array(rd.sample(train_input_PM2_5.station_id.unique(), 2))
sindex = np.where((train_input_PM2_5.station_id == stations_out[0]))[0]

X = train_input_PM2_5.drop(sindex).copy()
Y = train_output_PM2_5.drop(sindex).TARGET
X_test = train_input_PM2_5.iloc[sindex,:].copy()
Y_test = train_output_PM2_5.iloc[sindex].TARGET
'''
X.drop(['station_id', 'ID'], axis=1, inplace=True)
X_test.drop(['station_id', 'ID'], axis=1, inplace=True)
# grad boost
grad_boost_PM2_5 = GradientBoostingRegressor(loss='ls', learning_rate=0.08, n_estimators=1000, subsample=0.6, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=8, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=0.8, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

grad_boost_PM2_5.fit(X,Y)
square_error_PM2_5_train = (grad_boost_PM2_5.predict(X)-Y)**2
square_error_PM2_5 = (grad_boost_PM2_5.predict(X_test)-Y_test)**2
print("MSE train PM2_5 = "+str(np.mean(square_error_PM2_5_train)))
print("MSE test PM2_5 = "+str(np.mean(square_error_PM2_5)))
# meilleur 3.48317110392  learning_rate=0.2, n_estimators=500 max_depth=10,


print("MSE train = "+str((np.sum(square_error_NO2_train)+np.sum(square_error_PM10_train)+np.sum(square_error_PM2_5_train))/(len(square_error_NO2_train) + len(square_error_PM10_train)+ len(square_error_PM2_5_train))))
print("MSE test = "+str((np.sum(square_error_NO2)+np.sum(square_error_PM10)+np.sum(square_error_PM2_5))/(len(square_error_NO2) + len(square_error_PM10)+ len(square_error_PM2_5))))

# sur le test
res_NO2 = pd.DataFrame(grad_boost_NO2.predict(test_input_NO2.iloc[:,1:]), index = test_input_NO2.iloc[:,0], columns = ['TARGET'])
res_PM10 = pd.DataFrame(grad_boost_PM10.predict(test_input_PM10.iloc[:,1:]),  index = test_input_PM10.iloc[:,0], columns = ['TARGET'])
res_PM2_5 = pd.DataFrame(grad_boost_PM2_5.predict(test_input_PM2_5.iloc[:,1:]), index =  test_input_PM2_5.iloc[:,0], columns = ['TARGET'])
result = pd.concat([res_NO2, res_PM10, res_PM2_5]).sort_index()
result['ID'] = result.index
result = result[['ID', 'TARGET']]
result.to_csv("F:\MVA\SW\Outputs\challenge_output_V17.csv" ,sep = ',', index=False)

# Saving models
import pickle
pickle.dump(grad_boost_NO2, open("F:\MVA\SW\Models\GB\GB_NO2_V10.dat", "wb"))
pickle.dump(grad_boost_PM10, open("F:\MVA\SW\Models\GB\GB_PM10_V10.dat", "wb"))
pickle.dump(grad_boost_PM2_5, open("F:\MVA\SW\Models\GB\GB_PM2_5_V10.dat", "wb"))
