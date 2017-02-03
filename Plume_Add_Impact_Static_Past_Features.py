# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd
from sklearn.linear_model import LinearRegression
'''
File where we try to see the impact for adding the past values of the static coefficients
'''

#j'ai recopie ton true dans Plume_Bases
Train_input = pd.read_csv('X_train.csv')
Test_input = pd.read_csv('X_test.csv')
Train_output = pd.read_csv('challenge_output_data_training_file_predict_air_quality_at_the_street_level.csv')

train_input = Train_input.copy()
test_input = Test_input.copy()
train_output = Train_output.copy()

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
train_input.loc[:, land_cols] = train_input.loc[:, land_cols].fillna( value=0)
### JE LE FAIS AUSSI POUR test_input
test_input.loc[:, land_cols] = train_input.loc[:, land_cols].fillna( value=0)
#cree toirs colonnes selon chaque polluants
train_input = pd.get_dummies(train_input, columns = ["pollutant"])

#a quoi sert de faire ca sachant qu'on va le supprimer apres coup?
train_input["station_id"] = train_input["station_id"].astype('category')

#Remove, ne doit pas intervenir dans la prediction
train_input.drop(['station_id', 'ID'], axis=1, inplace=True)
test_input.drop(['station_id', 'ID'], axis=1, inplace=True)




#Seuls "pressure","windspeed","temperature","cloudcover","windbearingsin" et "windbearingcos" 
#varient significativement d'un temps à l'autre. Ce sont des parametres de meteo.

#on fait par polluant parce que sinon l'indice -1 est chiant
train_input_NO2 = train_input[train_input.pollutant_NO2 == 1].copy().reset_index()
train_input_PM10 = train_input[train_input.pollutant_PM10 == 1].copy().reset_index()
train_input_PM2_5 = train_input[train_input.pollutant_PM2_5 == 1].copy().reset_index()

train_output_NO2 = train_output[train_input.pollutant_NO2 == 1].copy().reset_index()
train_output_PM10 = train_output[train_input.pollutant_PM10 == 1].copy().reset_index()
train_output_PM2_5 = train_output[train_input.pollutant_PM2_5 == 1].copy().reset_index()

#je ne sais pas a quoi ca sert? C'est pour virer la case du pollutant_NO2?

train_input_NO2.drop('index', axis=1, inplace=True)
train_input_PM10.drop('index', axis=1, inplace=True)
train_input_PM2_5.drop('index', axis=1, inplace=True)

#regardons si sur N02 on prenant comme variable explicative le temps -1 de quelque variable explicatives on fait mieux que 
#sans les prendre. Après il faudra se demander jusqu'à combien de temps en arriere on ira.

def TabPast(data_Frame,Name):
    n= len(data_Frame.loc[:,Name])
    tabInter= data_Frame.loc[:,Name].values
    tab=np.ones(n)
    tab[1:n]=tabInter[0:(n-1)]
    tab[0]=tabInter[0]
    return(tab)


train_input_NO2.loc[:,'pressurePast']=pd.Series(TabPast(train_input_NO2,"pressure"),index=train_input_NO2.index)
train_input_NO2.loc[:,'temperaturePast']=pd.Series(TabPast(train_input_NO2,"temperature"),index=train_input_NO2.index)
train_input_NO2.loc[:,'windbearingsinPast']=pd.Series(TabPast(train_input_NO2,"windbearingsin"),index=train_input_NO2.index)
train_input_NO2.loc[:,'windbearingcosPast']=pd.Series(TabPast(train_input_NO2,"windbearingcos"),index=train_input_NO2.index)
#Seuls "pressure","windspeed","temperature","cloudcover","windbearingsin" et "windbearingcos" 
#varient significativement d'un temps à l'autre. Ce sont des parametres de meteo.

rindex = np.array(rd.sample(train_input_NO2.index, np.int(0.7*len(train_input_NO2))))
X = train_input_NO2.iloc[rindex,:]
Y = train_output_NO2.iloc[rindex].TARGET
X_test = train_input_NO2.iloc[~rindex,:]
Y_test = train_output_NO2.iloc[~rindex].TARGET
# Lin reg
linreg_NO2 = sk.linear_model.LinearRegression()
linreg_NO2.fit(X,Y)
square_error_NO2_train = (linreg_NO2.predict(X)-Y)**2
square_error_NO2 = (linreg_NO2.predict(X_test)-Y_test)**2
print("MSE train NO2 = "+str(np.mean(square_error_NO2_train)))
print("MSE test NO2 = "+str(np.mean(square_error_NO2)))





