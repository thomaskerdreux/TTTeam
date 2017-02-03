# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd
from sklearn.linear_model import LinearRegression

'''
This file aims at providing new feature for predicting. Namely the present, futur and past 
mean of the pollutant in a given city. But also the variance etc...
The second stage will be to do a  weighted mean. The weights will depend on the static 
coefficients of the stations.
'''


Train_input = pd.read_csv('X_train.csv')
Test_input = pd.read_csv('X_test.csv')
Train_output = pd.read_csv('challenge_output_data_training_file_predict_air_quality_at_the_street_level.csv')


train_input = Train_input.copy()
test_input = Test_input.copy()
train_output = Train_output.copy()


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



def score_function(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)/float(y_true.shape[0])
    
'''
Pour ajouter une colonne Y_mean
'''
#on selectionne les trois colonnes qui nous interessent
train_input_inter=train_input.loc[:,['pollutant','zone_id','daytime']]
train_input_inter['Pollutant_Mean']=pd.Series(train_output.loc[:,'TARGET'].values,index=train_input_inter.index)

Inter_Mean=train_input_inter.groupby(['daytime','zone_id','pollutant']).mean()
#on repasse les index dans des colonnes
Inter_Mean=Inter_Mean.reset_index(level=['daytime', 'zone_id','pollutant'])

#maintenant on merge train_input et Inter_Mean 
train_input_merge=pd.merge(train_input,Inter_Mean,how='inner', on= ['daytime', 'zone_id', 'pollutant'])
test_input_merge=pd.merge(test_input,Inter_Mean, how='inner', on= ['daytime', 'zone_id', 'pollutant'])
#ca a mis le bordel dans les indices
train_input_merge=train_input_merge.sort_values(by='ID', axis=0)
test_input_merge=test_input_merge.sort_values(by='ID', axis=0)



'''
POur ajouter une colonne Var_mean
'''

#on selectionne les trois colonnes qui nous interessent
train_input_bis=train_input.loc[:,['pollutant','zone_id','daytime']]
train_input_bis['Pollutant_Variance']=pd.Series(train_output.loc[:,'TARGET'].values,index=train_input_inter.index)


Inter_Var=train_input_bis.groupby(['daytime','zone_id','pollutant']).std()
#on repasse les index dans des colonnes
Inter_Var=Inter_Var.reset_index(level=['daytime', 'zone_id','pollutant'])

#maintenant on merge train_input et Inter_Var
train_input_merge_final=pd.merge(train_input_merge,Inter_Var, on= ['daytime', 'zone_id', 'pollutant'])
test_input_merge_final=pd.merge(test_input_merge,Inter_Var, how='inner', on= ['daytime', 'zone_id', 'pollutant'])
#ca a mis le bordel dans les indices

#c'est pas la bonne facon de sort le
train_input_merge_final=train_input_merge_final.sort_values(by='ID', axis=0)
test_input_merge_final=test_input_merge_final.sort_values(by='ID', axis=0)


'''
Pour ajouter Y_mean au temps -1, au temps +1. Il y a des problemes d'effets de bord.
Ils sont très graves si l'on decale juste la colonne Y_mean de 1 ou -1 sans separer pollutant
En revanche si l'on se place par pollutant, il y aura des effets de bords juste entre les stations
'''
##ON va pas trainer dix ans train_input_merge_final
train_input=train_input_merge_final
test_input=test_input_merge_final

train_input['Pollutant_Mean_Past']=pd.Series(train_input.loc[:,'Pollutant_Mean'].values,index=train_input.index)
test_input['Pollutant_Mean_Past']=pd.Series(test_input.loc[:,'Pollutant_Mean'].values,index=test_input.index)
train_input['Pollutant_Mean_Future']=pd.Series(train_input.loc[:,'Pollutant_Mean'].values,index=train_input.index)
test_input['Pollutant_Mean_Future']=pd.Series(test_input.loc[:,'Pollutant_Mean'].values,index=test_input.index)

def TabPast(data_Frame,Name,time):
    '''Petite fonction pour recuperer la tab décalée d'un indice/ facon plus short de faire?
    Il n'y aura qu'a modifier cette fonction pour mettre plus de variables explicatives
    attention que time doit prendre la valeur + ou - 1'''
    n= len(data_Frame.loc[:,Name])
    tabInter= data_Frame.loc[:,Name].values
    if n==0:
        print 'Not possible'
    tab=np.ones(n)
    if time==-1:
        tab[1:n]=tabInter[0:(n-1)]
        tab[0]=tabInter[0]
    else: #case time ==-1
        tab[0:(n-1)]=tabInter[1:n]
        tab[n-1]=tabInter[n-1]
    # mettre une erreur si l'on n'entre pas 1 ou -1 pour time
    return(tab)

tab_pollutant=['NO2', 'PM10', 'PM2_5' ]
#tableau de toutes les stations pour train et test
tab_station_train=train_input.loc[:,'station_id'].value_counts().index
tab_station_test=test_input.loc[:,'station_id'].value_counts().index
tab_station=np.concatenate((tab_station_train,tab_station_test))
'''
Peut etre faut-il rajouter une colonne a train_input
'''
for i in range(3):
    for j in range(tab_station_train.size):
        M_train=train_input[(train_input.station_id==tab_station[j]) & (train_input.pollutant==tab_pollutant[i])].size
        M_test=test_input[(test_input.station_id==tab_station[j]) & (test_input.pollutant==tab_pollutant[i])].size
        if (j<tab_station_train.size) & (M_train !=0 ):
            #on recupere le tableau decale d'un indice 
            Frame_inter=train_input[(train_input.station_id==tab_station[j]) & (train_input.pollutant==tab_pollutant[i])]
            tab_inter_past=TabPast(Frame_inter,'Pollutant_Mean_Past',-1)
            tab_inter_future=TabPast(Frame_inter,'Pollutant_Mean_Future',1)
            Frame_inter['Pollutant_Mean_Past']=pd.Series(tab_inter_past,index=Frame_inter.index)
            Frame_inter['Pollutant_Mean_Future']=pd.Series(tab_inter_future,index=Frame_inter.index)
            train_input.update(Frame_inter)
            print(tab_station_train[j])
            print(tab_pollutant[i])
        elif (M_test !=0 ):
            Frame_inter=test_input[(test_input.station_id==tab_station[j]) & (test_input.pollutant==tab_pollutant[i])]
            tab_inter_past=TabPast(Frame_inter,'Pollutant_Mean_Past',-1)
            tab_inter_future=TabPast(Frame_inter,'Pollutant_Mean_Future',1)
            Frame_inter['Pollutant_Mean_Past']=pd.Series(tab_inter_past,index=Frame_inter.index)
            Frame_inter['Pollutant_Mean_Future']=pd.Series(tab_inter_future,index=Frame_inter.index)
            test_input.update(Frame_inter)
            print(tab_station_train[j])
            print(tab_pollutant[i])
    



'''
On exporte tout dans un nouveau fichier. Pour l'instant on a ajouté:
Pollutant_Mean ; Pollutant_Variance ; Pollutant_Mean_Past ; Pollutant_Mean_Future
'''
train_input=train_input.sort_values(by='ID', axis=0)
test_input=test_input.sort_values(by='ID', axis=0)

train_input.to_csv("train_input_add_feature.csv" ,sep = ',', index=False)
test_input.to_csv("test_input_add_feature.csv" ,sep = ',', index=False)

