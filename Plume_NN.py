''' Recupérer données '''
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd

fraction = 0.7

''' scaler les données '''
from sklearn.preprocessing import StandardScaler 


# NO2
'''
# Selection train-test par station
stations_out = np.array(rd.sample(train_input_NO2.station_id.unique(), 4))
sindex = np.where((train_input_NO2.station_id == stations_out[0]) | (train_input_NO2.station_id == stations_out[1]) | (train_input_NO2.station_id == stations_out[2]) | (train_input_NO2.station_id == stations_out[3]))[0]

X = train_input_NO2.drop(sindex).copy()
Y = train_output_NO2.drop(sindex).TARGET
X_test = train_input_NO2.iloc[sindex,:].copy()
Y_test = train_output_NO2.iloc[sindex].TARGET
'''

''' NO2 '''
# Selection train-test par rows
rindex = np.array(rd.sample(train_input_NO2.index, np.int(0.7*len(train_input_NO2))))
X = train_input_NO2.iloc[rindex,:].copy()
Y = train_output_NO2.iloc[rindex].TARGET
X_test = train_input_NO2.drop(rindex).copy()
Y_test = train_output_NO2.drop(rindex).TARGET

X.drop(['station_id', 'ID'], axis=1, inplace=True)
X_test.drop(['station_id', 'ID'], axis=1, inplace=True)


scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X)  
X = scaler.transform(X)  
# apply same transformation to test data
X_test = scaler.transform(X_test) 
# On real test data
X_test_NO2 = scaler.transform(test_input_NO2.iloc[:,1:])

from sklearn.neural_network import MLPRegressor

NN_NO2 = MLPRegressor(hidden_layer_sizes=(1000, 1000, 1000, 1000), activation='relu', solver='adam', alpha=0.01, batch_size='auto',
                               learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, max_iter=300, shuffle=True,
                               random_state=None, tol=0.001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                               early_stopping=True, validation_fraction=0.15, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
NN_NO2.fit(X, Y)
square_error_NO2_train = (NN_NO2.predict(X)-Y)**2
square_error_NO2 = (NN_NO2.predict(X_test)-Y_test)**2
print("MSE train NO2 = "+str(np.mean(square_error_NO2_train)))
print("MSE test NO2 = "+str(np.mean(square_error_NO2)))

''' PM10 '''
# Selection train-test par rows
rindex = np.array(rd.sample(train_input_PM10.index, np.int(0.7*len(train_input_PM10))))
X = train_input_PM10.iloc[rindex,:].copy()
Y = train_output_PM10.iloc[rindex].TARGET
X_test = train_input_PM10.drop(rindex).copy()
Y_test = train_output_PM10.drop(rindex).TARGET

X.drop(['station_id', 'ID'], axis=1, inplace=True)
X_test.drop(['station_id', 'ID'], axis=1, inplace=True)

scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X)  
X = scaler.transform(X)  
# apply same transformation to test data
X_test = scaler.transform(X_test) 
# On real test data
X_test_PM10 = scaler.transform(test_input_PM10.iloc[:,1:])

from sklearn.neural_network import MLPRegressor

NN_PM10 = MLPRegressor(hidden_layer_sizes=(1000, 1000, 1000, 1000), activation='relu', solver='adam', alpha=0.01, batch_size='auto',
                               learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, max_iter=300, shuffle=True,
                               random_state=None, tol=0.001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                               early_stopping=True, validation_fraction=0.15, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

NN_PM10.fit(X, Y)
square_error_PM10_train = (NN_PM10.predict(X)-Y)**2
square_error_PM10 = (NN_PM10.predict(X_test)-Y_test)**2
print("MSE train PM10 = "+str(np.mean(square_error_PM10_train)))
print("MSE test PM10 = "+str(np.mean(square_error_PM10)))

''' PM2_5 '''
# Selection train-test par rows
rindex = np.array(rd.sample(train_input_PM2_5.index, np.int(0.7*len(train_input_PM2_5))))
X = train_input_PM2_5.iloc[rindex,:].copy()
Y = train_output_PM2_5.iloc[rindex].TARGET
X_test = train_input_PM2_5.drop(rindex).copy()
Y_test = train_output_PM2_5.drop(rindex).TARGET

X.drop(['station_id', 'ID'], axis=1, inplace=True)
X_test.drop(['station_id', 'ID'], axis=1, inplace=True)

scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(X)  
X = scaler.transform(X)  
# apply same transformation to test data
X_test = scaler.transform(X_test) 
# On real test data
X_test_PM2_5 = scaler.transform(test_input_PM2_5.iloc[:,1:])

from sklearn.neural_network import MLPRegressor

NN_PM2_5 = MLPRegressor(hidden_layer_sizes=(1000, 1000, 1000, 1000), activation='relu', solver='adam', alpha=0.01, batch_size='auto',
                               learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, max_iter=300, shuffle=True,
                               random_state=None, tol=0.001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                               early_stopping=True, validation_fraction=0.15, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

NN_PM2_5.fit(X, Y)
square_error_PM2_5_train = (NN_PM2_5.predict(X)-Y)**2
square_error_PM2_5 = (NN_PM2_5.predict(X_test)-Y_test)**2
print("MSE train PM2_5 = "+str(np.mean(square_error_PM2_5_train)))
print("MSE test PM2_5 = "+str(np.mean(square_error_PM2_5)))

print("MSE train = "+str((np.sum(square_error_NO2_train)+np.sum(square_error_PM10_train)+np.sum(square_error_PM2_5_train))/(len(square_error_NO2_train) + len(square_error_PM10_train)+ len(square_error_PM2_5_train))))
print("MSE test = "+str((np.sum(square_error_NO2)+np.sum(square_error_PM10)+np.sum(square_error_PM2_5))/(len(square_error_NO2) + len(square_error_PM10)+ len(square_error_PM2_5))))

# sur le test
res_NO2 = pd.DataFrame(NN_NO2.predict(X_test_NO2), index = test_input_NO2.iloc[:,0], columns = ['TARGET'])
res_PM10 = pd.DataFrame(NN_PM10.predict(X_test_PM10),  index = test_input_PM10.iloc[:,0], columns = ['TARGET'])
res_PM2_5 = pd.DataFrame(NN_PM2_5.predict(X_test_PM2_5), index =  test_input_PM2_5.iloc[:,0], columns = ['TARGET'])
result = pd.concat([res_NO2, res_PM10, res_PM2_5]).sort_index()
result['ID'] = result.index
result = result[['ID', 'TARGET']]
result.to_csv("F:\MVA\SW\Outputs\challenge_output_V16.csv" ,sep = ',', index=False)

# Saving models
import pickle
pickle.dump(NN_NO2, open("F:\MVA\SW\Models\NN\MLP_NO2_V2.dat", "wb"))
pickle.dump(NN_PM10, open("F:\MVA\SW\Models\NN\MLP_PM10_V2.dat", "wb"))
pickle.dump(NN_PM2_5, open("F:\MVA\SW\Models\NN\MLP_PM2_5_V2.dat", "wb"))