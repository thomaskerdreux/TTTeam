# -*- coding: utf-8 -*-
"""
This file aims at trying very simple regression methods to validate the import of new features.
With simple regression.
 Pollutant_Mean ; Pollutant_Variance ; Pollutant_Mean_Past ; Pollutant_Mean_Future
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd
from sklearn.linear_model import LinearRegression

'''Loading from feature add data sets'''
Train_input = pd.read_csv('train_input_add_feature.csv')
Test_input = pd.read_csv('test_input_add_feature.csv')
Train_output = pd.read_csv('challenge_output_data_training_file_predict_air_quality_at_the_street_level.csv')

train_input = Train_input.copy()
test_input = Test_input.copy()
train_output = Train_output.copy()

train_input = pd.get_dummies(train_input, columns = ["pollutant"])

''' Score '''
def score_function(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)/float(y_true.shape[0])
    
''' Traitement '''
# Traitement train_input
#on reindexe les label par ordre alphabetique
train_input = train_input.reindex_axis(sorted(train_input.columns), axis=1)
land_cols = ["hlres_1000","hlres_500","hlres_300", "hlres_100", "hlres_50", "hldres_1000","hldres_500","hldres_300", "hldres_100",
             "hldres_50", "industry_1000", "route_1000", "route_500", "route_300", "route_100", "port_5000", "natural_5000",
            "green_5000"]
#remplace toute les valeurs non-existantes par zero: Est-ce une bonne idée?
train_input.loc[:, :] = train_input.loc[:, :].fillna( value=0)
test_input.loc[:, :] = train_input.loc[:, :].fillna( value=0)

#a quoi sert de faire ca sachant qu'on va le supprimer apres coup?
train_input["station_id"] = train_input["station_id"].astype('category')

#Remove, ne doit pas intervenir dans la prediction
train_input.drop(['station_id', 'ID'], axis=1, inplace=True)
test_input.drop(['station_id', 'ID'], axis=1, inplace=True)
    


''' Regression linéaire avec tous les nouveaux features'''

rindex = np.array(rd.sample(train_input.index, np.int(0.7*len(train_input))))

X = train_input.iloc[rindex,:]
Y = train_output.loc[rindex,"TARGET"]
X_test = train_input.iloc[~rindex,:]
Y_test = train_output.iloc[~rindex].TARGET

linreg = sk.linear_model.LinearRegression()
linreg.fit(X,Y)
print('MSE train = ' +str(score_function(Y, linreg.predict(X))))
print('MSE test = ' +str(score_function(Y_test, linreg.predict(X_test))))




''' Regression linéaire en retirant Pollutant_Variance'''
train_input.drop(['Pollutant_Variance'], axis=1, inplace=True)
test_input.drop(['Pollutant_Variance'], axis=1, inplace=True)

rindex = np.array(rd.sample(train_input.index, np.int(0.7*len(train_input))))

X = train_input.iloc[rindex,:]
Y = train_output.loc[rindex,"TARGET"]
X_test = train_input.iloc[~rindex,:]
Y_test = train_output.iloc[~rindex].TARGET

linreg = sk.linear_model.LinearRegression()
linreg.fit(X,Y)
print('MSE train = ' +str(score_function(Y, linreg.predict(X))))
print('MSE test = ' +str(score_function(Y_test, linreg.predict(X_test))))

''' Regression linéaire en retirant Pollutant_Variance et Pollutant_Mean_Future'''
train_input.drop(['Pollutant_Mean_Future'], axis=1, inplace=True)
test_input.drop(['Pollutant_Mean_Future'], axis=1, inplace=True)

rindex = np.array(rd.sample(train_input.index, np.int(0.7*len(train_input))))

X = train_input.iloc[rindex,:]
Y = train_output.loc[rindex,"TARGET"]
X_test = train_input.iloc[~rindex,:]
Y_test = train_output.iloc[~rindex].TARGET

linreg = sk.linear_model.LinearRegression()
linreg.fit(X,Y)
print('MSE train = ' +str(score_function(Y, linreg.predict(X))))
print('MSE test = ' +str(score_function(Y_test, linreg.predict(X_test))))


''' Regression linéaire en retirant Pollutant_Variance et Pollutant_Mean_Future et Pollutant_Mean_Past '''
train_input.drop(['Pollutant_Mean_Past'], axis=1, inplace=True)
test_input.drop(['Pollutant_Mean_Past'], axis=1, inplace=True)

rindex = np.array(rd.sample(train_input.index, np.int(0.7*len(train_input))))

X = train_input.iloc[rindex,:]
Y = train_output.loc[rindex,"TARGET"]
X_test = train_input.iloc[~rindex,:]
Y_test = train_output.iloc[~rindex].TARGET

linreg = sk.linear_model.LinearRegression()
linreg.fit(X,Y)
print('MSE train = ' +str(score_function(Y, linreg.predict(X))))
print('MSE test = ' +str(score_function(Y_test, linreg.predict(X_test))))


''' Regression linéaire en retirant toutes les add_features'''
train_input.drop(['Pollutant_Mean'], axis=1, inplace=True)
test_input.drop(['Pollutant_Mean'], axis=1, inplace=True)

rindex = np.array(rd.sample(train_input.index, np.int(0.7*len(train_input))))

X = train_input.iloc[rindex,:]
Y = train_output.loc[rindex,"TARGET"]
X_test = train_input.iloc[~rindex,:]
Y_test = train_output.iloc[~rindex].TARGET

linreg = sk.linear_model.LinearRegression()
linreg.fit(X,Y)
print('MSE train = ' +str(score_function(Y, linreg.predict(X))))
print('MSE test = ' +str(score_function(Y_test, linreg.predict(X_test))))
