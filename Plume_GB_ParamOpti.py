''' Recupérer données '''
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd
from sklearn.ensemble import GradientBoostingRegressor

fraction = 0.7

Max_depth = [10]
Learning_rate = [0.1, 0.2, 0.5]
Subsample = [1., 0.8, 0.6]
Max_features = [1., 0.8, 0.6]

Square_error_NO2 = np.zeros((1,3,3,3))
Square_error_PM10 = np.zeros((1,3,3,3))
Square_error_PM2_5 = np.zeros((1,3,3,3))
Global_square_error = np.zeros((1,3,3,3))

#for max_depth, learning_rate, subsample, max_features in [(max_depth, learning_rate, subsample, max_features) for max_depth in Max_depth for learning_rate in Learning_rate for subsample in Subsample for max_features in Max_features]:
#    print max_depth, learning_rate, subsample, max_features 
    
for i, j, k, l, max_depth, learning_rate, subsample, max_features in [(i, j, k, l, max_depth, learning_rate, subsample, max_features) for (i, max_depth) in enumerate(Max_depth) for (j,learning_rate) in enumerate(Learning_rate) for (k,subsample) in enumerate(Subsample) for (l, max_features) in enumerate(Max_features)]:
    print i, j, k, l, max_depth, learning_rate, subsample, max_features 

    '''Triple Grad Boost '''
    rindex = np.array(rd.sample(train_input_NO2.index, np.int(0.7*len(train_input_NO2))))
    X = train_input_NO2.iloc[rindex,:].copy()
    Y = train_output_NO2.iloc[rindex].TARGET
    X_test = train_input_NO2.drop(rindex).copy()
    Y_test = train_output_NO2.drop(rindex).TARGET
    
    X.drop(['station_id', 'ID'], axis=1, inplace=True)
    X_test.drop(['station_id', 'ID'], axis=1, inplace=True)

    # grad boost
    grad_boost_NO2 = GradientBoostingRegressor(loss='ls', learning_rate=learning_rate, n_estimators=50, subsample=subsample, criterion='friedman_mse',
                                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=max_depth, 
                                           min_impurity_split=1e-07, init=None, random_state=None, max_features=max_features, alpha=0.9, 
                                           verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

    grad_boost_NO2.fit(X,Y)
    square_error_NO2_train = (grad_boost_NO2.predict(X)-Y)**2
    square_error_NO2 = (grad_boost_NO2.predict(X_test)-Y_test)**2
    Square_error_NO2[i,j,k,l] = np.mean(square_error_NO2)
    print("MSE test NO2 = "+str(np.mean(square_error_NO2)))
    # meilleur 35.8153639899 learning_rate=0.1, n_estimators=500 max_depth=10,

    rindex = np.array(rd.sample(train_input_PM10.index, np.int(fraction*len(train_input_PM10))))
    X = train_input_PM10.iloc[rindex,:].copy()
    Y = train_output_PM10.iloc[rindex].TARGET
    X_test = train_input_PM10.drop(rindex).copy()
    Y_test = train_output_PM10.drop(rindex).TARGET
    
    X.drop(['station_id', 'ID'], axis=1, inplace=True)
    X_test.drop(['station_id', 'ID'], axis=1, inplace=True)

    # grad boost
    grad_boost_PM10 =  GradientBoostingRegressor(loss='ls', learning_rate=learning_rate, n_estimators=50, subsample=subsample, criterion='friedman_mse',
                                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=max_depth, 
                                           min_impurity_split=1e-07, init=None, random_state=None, max_features=max_features, alpha=0.9, 
                                           verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    grad_boost_PM10.fit(X,Y)
    square_error_PM10_train = (grad_boost_PM10.predict(X)-Y)**2
    square_error_PM10 = (grad_boost_PM10.predict(X_test)-Y_test)**2
    Square_error_PM10[i,j,k,l] = np.mean(square_error_PM10)
    print("MSE test PM10 = "+str(np.mean(square_error_PM10)))
    # meilleur 13.3930515372 learning_rate=0.2, n_estimators=500 max_depth=10,

    rindex = np.array(rd.sample(train_input_PM2_5.index, np.int(0.7*len(train_input_PM2_5))))
    X = train_input_PM2_5.iloc[rindex,:].copy()
    Y = train_output_PM2_5.iloc[rindex].TARGET
    X_test = train_input_PM2_5.drop(rindex).copy()
    Y_test = train_output_PM2_5.drop(rindex).TARGET
    
    X.drop(['station_id', 'ID'], axis=1, inplace=True)
    X_test.drop(['station_id', 'ID'], axis=1, inplace=True)

    # grad boost
    grad_boost_PM2_5 =  GradientBoostingRegressor(loss='ls', learning_rate=learning_rate, n_estimators=50, subsample=subsample, criterion='friedman_mse',
                                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=max_depth, 
                                           min_impurity_split=1e-07, init=None, random_state=None, max_features=max_features, alpha=0.9, 
                                           verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

    grad_boost_PM2_5.fit(X,Y)
    square_error_PM2_5_train = (grad_boost_PM2_5.predict(X)-Y)**2
    square_error_PM2_5 = (grad_boost_PM2_5.predict(X_test)-Y_test)**2
    Square_error_PM2_5[i,j,k,l] = np.mean(square_error_PM2_5)
    print("MSE test PM2_5 = "+str(np.mean(square_error_PM2_5)))
    # meilleur 3.48317110392  learning_rate=0.2, n_estimators=500 max_depth=10,

    Global_square_error[i,j,k,l] = (np.sum(square_error_NO2)+np.sum(square_error_PM10)+np.sum(square_error_PM2_5))/(len(square_error_NO2) + len(square_error_PM10)+ len(square_error_PM2_5))
    print("MSE train = "+str((np.sum(square_error_NO2_train)+np.sum(square_error_PM10_train)+np.sum(square_error_PM2_5_train))/(len(square_error_NO2_train) + len(square_error_PM10_train)+ len(square_error_PM2_5_train))))
    print("MSE test = "+str((np.sum(square_error_NO2)+np.sum(square_error_PM10)+np.sum(square_error_PM2_5))/(len(square_error_NO2) + len(square_error_PM10)+ len(square_error_PM2_5))))

np.save("F:\MVA\SW\Models\GB\ParamOpti\Square_error_NO2.npy" , Square_error_NO2 )
np.save("F:\MVA\SW\Models\GB\ParamOpti\Square_error_PM10.npy" , Square_error_PM10 )
np.save("F:\MVA\SW\Models\GB\ParamOpti\Square_error_PM2_5.npy" , Square_error_PM2_5 )
np.save("F:\MVA\SW\Models\GB\ParamOpti\Global_square_error.npy" , Global_square_error )

'''
Square_error_NO2 = np.load("F:\MVA\SW\Models\GB\ParamOpti\Square_error_NO2.npy")[0, :, :, :]
Square_error_PM10 = np.load("F:\MVA\SW\Models\GB\ParamOpti\Square_error_PM10.npy" )[0, :, :, :]
Square_error_PM2_5 = np.load("F:\MVA\SW\Models\GB\ParamOpti\Square_error_PM2_5.npy"  )[0, :, :, :]
Global_square_error = np.load("F:\MVA\SW\Models\GB\ParamOpti\Global_square_error.npy"  )[0, :, :, :]

'''
