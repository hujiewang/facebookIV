__author__ = 'hujie'


import pandas as pd
import numpy as np
import data as data
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import getopt
import sys
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

random.seed()
train_x,train_y,valid_x,valid_y,test_x,valid_id,test_id=data.loadData()

def train(model_id,train_x,train_y,valid_x,valid_y,test_x):
    train_x,train_y=shuffle(train_x,train_y)


    random_state=random.randint(0, 1000000)
    rf = RandomForestClassifier(n_jobs=8)

    param_dist = {
            "n_estimators":sp_randint(100,300),
        "criterion": ["gini"],
        #"max_depth": sp_randint(3, 10000),
        #"min_samples_split": sp_randint(1, 300),
        #"min_samples_leaf": sp_randint(1, 300),
        "max_features": sp_randint(10, 26),
        "bootstrap": [True, False],
        'random_state':sp_randint(1, 1000000),
        }

    clf = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=50,cv=10,scoring='roc_auc')

    clf.fit(train_x, train_y)
    valid_predictions = clf.predict_proba(valid_x)[:, 1]
    test_predictions= clf.predict_proba(test_x)[:, 1]

    loss = roc_auc_score(valid_y,valid_predictions)
    print('loss:')
    print(loss)
    print(clf.best_estimator_)
    data.saveData(valid_id,valid_predictions,"./valid_results/valid_"+str(model_id)+".csv")
    data.saveData(test_id,test_predictions,"./results/results_"+str(model_id)+".csv")



train(5,train_x,train_y,valid_x,valid_y,test_x)