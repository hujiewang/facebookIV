__author__ = 'hujie'
import pandas as pd
import numpy as np

def loadData():
    train_x=pd.read_csv("./data/train_x.csv")
    train_y=pd.read_csv("./data/train_y.csv")
    valid_x=pd.read_csv("./data/valid_x.csv")
    valid_y=pd.read_csv("./data/valid_y.csv")
    test_x=pd.read_csv("./data/test_x.csv")

    valid_id=pd.read_csv("./data/valid_id.csv")
    test_id=pd.read_csv("./data/test_id.csv")

    train_y=train_y.values
    _train_y=[]
    for i in range(len(train_y)):
        _train_y.append(train_y[i,0])
    valid_y=valid_y.values
    _valid_y=[]
    for i in range(len(valid_y)):
        _valid_y.append(valid_y[i,0])

    return train_x.values,np.array(_train_y),valid_x.values,np.array(_valid_y),test_x.values,valid_id.values,test_id.values

def saveData(id,predictions,fpath):
    data=np.hstack((id,predictions.astype(str).reshape(len(predictions),1)))
    np.savetxt(
    fpath, data, fmt="%s,%s", delimiter=",", header="bidder_id,prediction", comments=''
    )
    print('Predictions has been saved!')