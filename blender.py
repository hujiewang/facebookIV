__author__ = 'hujie'

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import os
import random
from tqdm import tqdm
import re
import data as util
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
# Load validation data set
train_x,train_y,valid_x,valid_y,test_x,valid_id,test_id=util.loadData()

# Load list of model results
predictions = []
results = []
fname = []
fname_rv = []

for file in os.listdir("./valid_results"):
    fname.append(file)
for file in os.listdir("./results"):
    fname_rv.append(file)
def getID(s):
    return int(re.findall(r'[\d]+',s)[0])

fname = sorted(fname, key=getID)
fname_rv = sorted(fname_rv, key=getID )

exclusion = set([1,3,4,5,6,7])
for f in fname:
    if getID(f) not in exclusion:
        data=pd.read_csv("./valid_results/"+f)
        _data=data[["prediction"]]
        predictions.append(_data.values)
        loss = roc_auc_score(valid_y,_data.values)
        print(f+" loss: "),
        print(loss)

for f in fname_rv:
    if getID(f) not in exclusion:
        data=pd.read_csv("./results/"+f)
        _data=data[["prediction"]]
        results.append(_data.values)


def loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return roc_auc_score(valid_y,final_prediction)

best_score= 0.0
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
bounds = [(0,1)]*len(predictions)

best_weights = [1.0/len(predictions)]*len(predictions)
for i in tqdm(range(0)):
    weights = [0.0]*len(predictions)
    for i in range(len(weights)):
        weights[i]=random.uniform(0, 1)

    res = minimize(loss_func, weights, method='SLSQP', bounds=bounds, constraints=cons)

    if res['fun']>best_score:
        best_score = res['fun']
        best_weights=res['x']

print('Best Ensamble Score: {best_score}'.format(best_score=best_score))
print('\nBlending models...')
print(best_weights)

final_pred = 0
for weight,result in zip(best_weights, predictions):
    final_pred += weight*result

loss = roc_auc_score(valid_y,final_pred)
print('loss:')
print(loss)

final_rv = 0
for weight,result in zip(best_weights, results):
    final_rv += weight*result

util.saveData(test_id,final_rv,"./final_results/final_results.csv")
print('Blended!')