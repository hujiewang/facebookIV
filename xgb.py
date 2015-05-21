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
from XGBoostClassifier import XGBoostClassifier
from tqdm import tqdm

random.seed()
train_x,train_y,valid_x,valid_y,test_x,valid_id,test_id=data.loadData()

def train(model_id,train_x,train_y,valid_x,valid_y,test_x):
    train_x,train_y=shuffle(train_x,train_y)

    maximum_auc=0.0
    random_state=random.randint(0, 1000000)
    for i in tqdm(range(1000)):
        config={
        'base_estimator':'gbtree',
        'objective':'multi:softprob',
        'metric':'mlogloss',
        'num_classes':2,
        'learning_rate':random.uniform(0.01,0.15),
        'max_depth':20+random.randint(0,10),
        'max_samples':random.uniform(0.3,1.0),
        'max_features':random.uniform(0.3,1.0),
        'max_delta_step':random.randint(1,10),
        'min_child_weight':random.randint(1,8),
        'min_loss_reduction':1,
        'l1_weight':random.uniform(0.0,10.0),
        'l2_weight':random.uniform(0.0,10.0),
        'l2_on_bias':False,
        'gamma':random.uniform(0.0,0.1),
        'inital_bias':0.5,
        'random_state':random_state,

        }
        clf = XGBoostClassifier(
            config['base_estimator'],
            config['objective'],
            config['metric'],
            config['num_classes'],
            config['learning_rate'],
            config['max_depth'],
            config['max_samples'],
            config['max_features'],
            config['max_delta_step'],
            config['min_child_weight'],
            config['min_loss_reduction'],
            config['l1_weight'],
            config['l2_weight'],
            config['l2_on_bias'],
            config['gamma'],
            config['inital_bias'],
            config['random_state'],
            watchlist=[[valid_x,valid_y]],
            n_jobs=8,
            n_iter=30000,
        )
        clf.fit(train_x, train_y)

        valid_predictions = clf.predict_proba(valid_x)[:, 1]
        test_predictions= clf.predict_proba(test_x)[:, 1]

        auc = roc_auc_score(valid_y,valid_predictions)
        if auc>maximum_auc:
            maximum_auc=auc
            best_config=config
            print('new auc:')
            print(auc)
            data.saveData(valid_id,valid_predictions,"./valid_results/valid_"+str(model_id)+".csv")
            data.saveData(test_id,test_predictions,"./results/results_"+str(model_id)+".csv")
    print('maximum_auc:')
    print(maximum_auc)
    print(config)
train(8,train_x,train_y,valid_x,valid_y,test_x)