import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import random as rd
from sklearn.ensemble import GradientBoostingRegressor


param_grid = [
  {'learning_rate': [0.1, 0.05], 'n_estimators': [200, 300,  500]}
 ]

                                     
                    
'''Triple Grad Boost '''
X = train_input_NO2
Y = train_output_NO2.TARGET
# grad boost
grad_boost_NO2 = GradientBoostingRegressor(loss='ls', subsample=0.6, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=0.7, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

cv_grid_NO2 = sk.model_selection.GridSearchCV(grad_boost_NO2, param_grid, scoring=None, fit_params=None, n_jobs=3, iid=True, refit=True, cv=3, verbose=0, 
                                     pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)
cv_grid_NO2.fit(X, Y)
X = train_input_PM10
Y = train_output_PM10.TARGET
# grad boost
grad_boost_PM10 = GradientBoostingRegressor(loss='ls', subsample=0.6, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=0.7, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
cv_grid_PM10 = sk.model_selection.GridSearchCV(grad_boost_PM10, param_grid, scoring=None, fit_params=None, n_jobs=3, iid=True, refit=True, cv=3, verbose=0, 
                                     pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)
print("PM10")
cv_grid_PM10.fit(X, Y)

X = train_input_PM2_5
Y = train_output_PM2_5.TARGET
# grad boost
grad_boost_PM2_5 = GradientBoostingRegressor(loss='ls', subsample=0.6, criterion='friedman_mse',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                       min_impurity_split=1e-07, init=None, random_state=None, max_features=0.7, alpha=0.9, 
                                       verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

cv_grid_PM2_5 = sk.model_selection.GridSearchCV(grad_boost_PM2_5, param_grid, scoring=None, fit_params=None, n_jobs=3, iid=True, refit=True, cv=3, verbose=0, 
                                     pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True)
print("PM2_5")
cv_grid_PM2_5.fit(X, Y)
