''' Recupérer données '''
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd


Train_input = pd.read_csv('F:\MVA\SW\X_train.csv')
Test_input = pd.read_csv('F:\MVA\SW\X_test.csv')
Train_output = pd.read_csv('F:\MVA\SW\challenge_output_data_training_file_predict_air_quality_at_the_street_level.csv')

train_input = Train_input.copy()
test_input = Test_input.copy()
train_output = Train_output.copy()

''' Score '''
def score_function(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)/float(y_true.shape[0])
    
''' Traitement '''
# Traitement input
train_input = train_input.reindex_axis(sorted(train_input.columns), axis=1)
test_input = test_input.reindex_axis(sorted(train_input.columns), axis=1)
land_cols = ["hlres_1000","hlres_500","hlres_300", "hlres_100", "hlres_50", "hldres_1000","hldres_500","hldres_300", "hldres_100",
             "hldres_50", "industry_1000", "route_1000", "route_500", "route_300", "route_100", "port_5000", "natural_5000",
            "green_5000"]

train_input.loc[:, land_cols] = train_input.loc[:, land_cols].fillna( value=0)
train_input = pd.get_dummies(train_input, columns = ["pollutant"])

train_input["station_id"] = train_input["station_id"].astype('category')

test_input.loc[:, land_cols] = test_input.loc[:, land_cols].fillna( value=0)
test_input = pd.get_dummies(test_input, columns = ["pollutant"])

test_input["station_id"] = test_input["station_id"].astype('category')

train_input.describe()

'''Décrire les stations '''
# On récupère le noms des colonnes qui nous intéressent : celles qui ont une variance non nulle dans au moins une station
# Note : il faut s'assurer que ce sont bien les même sur le train et le test

# Note2 : checker "daytime"

variable_col = (train_input.groupby("station_id", as_index=False).var().reindex_axis(sorted(train_input.columns), axis=1)!=0).any(axis = 0) 
variable_col.loc["ID"] = 0

# On calcule des grandeur par station
Means = train_input.loc[:, variable_col].groupby("station_id", as_index = False).mean()
Var = train_input.loc[:, variable_col].groupby("station_id", as_index = False).var()
Med = train_input.loc[:, variable_col].groupby("station_id", as_index = False).median()
Quant = train_input.loc[:, variable_col].groupby("station_id", as_index = False).quantile(q = 0.70)
# On peut rajouter des moments d'ordre supérieurs, des quantiles...

# On merge par station_id avec la base initiale
train_input = pd.merge(train_input, Means, suffixes=('', '_mean'),  on = 'station_id')
train_input = pd.merge(train_input, Var, suffixes=('', '_var'), on ="station_id")
#train_input = pd.merge(train_input, Med, suffixes=('', '_med'), on  ="station_id")


# On calcule des grandeur par station
Means = test_input.loc[:, variable_col].groupby("station_id", as_index = False).mean()
Var = test_input.loc[:, variable_col].groupby("station_id", as_index = False).var()
Med = test_input.loc[:, variable_col].groupby("station_id", as_index = False).median()
Quant = test_input.loc[:, variable_col].groupby("station_id", as_index = False).quantile(q = 0.70)
# On peut rajouter des moments d'ordre supérieurs, des quantiles...

# On merge par station_id avec la base initiale
test_input = pd.merge(test_input, Means, suffixes=('', '_mean'),  on = 'station_id')
test_input = pd.merge(test_input, Var, suffixes=('', '_var'), on ="station_id")
#train_input = pd.merge(train_input, Med, suffixes=('', '_med'), on  ="station_id")

#Remove, ne doit pas intervenir dans la prediction
#train_input.drop(['station_id', 'ID'], axis=1, inplace=True)
test_input.drop(['station_id', 'ID'], axis=1, inplace=True)


''' Séparer les polluants '''

train_input_NO2 = train_input[train_input.pollutant_NO2 == 1].copy().reset_index()
train_input_PM10 = train_input[train_input.pollutant_PM10 == 1].copy().reset_index()
train_input_PM2_5 = train_input[train_input.pollutant_PM2_5 == 1].copy().reset_index()

train_output_NO2 = train_output[train_input.pollutant_NO2 == 1].copy().reset_index()
train_output_PM10 = train_output[train_input.pollutant_PM10 == 1].copy().reset_index()
train_output_PM2_5 = train_output[train_input.pollutant_PM2_5 == 1].copy().reset_index()


train_input_NO2.drop('index', axis=1, inplace=True)
train_input_PM10.drop('index', axis=1, inplace=True)
train_input_PM2_5.drop('index', axis=1, inplace=True)

# Test
test_input_NO2 = test_input[test_input.pollutant_NO2 == 1].copy().reset_index()
test_input_PM10 = test_input[test_input.pollutant_PM10 == 1].copy().reset_index()
test_input_PM2_5 = test_input[test_input.pollutant_PM2_5 == 1].copy().reset_index()


