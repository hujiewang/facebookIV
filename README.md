# kaggle-facebookIV
# current ranking: 96th

## Classifier algorithms 
* NN(TODO)
* XGB
* Random Forests

## Feature extraction methods  
* Will be available after the contest

## Software
* python 2.7
* numpy
* scipy
* scikit-learn 
* xgboost

## Usage
* Put all data set in ./data 
* python featEng.py (Currently hidden)
    * Output files: train_x.csv,train_y.csv,valid_x.csv,valid_y.csv,test_x.csv,valid_id.csv,test_id.csv
* python rf.py (Random Forest model)
    * Output files: 
    * ./valid_results/valid_[model_id].csv (validation dataset)
    * ./results/results_[model_id].csv  (test dataset)
* python xgb.py (XGB model)
    * Output files: 
    * ./valid_results/valid_[model_id].csv (validation dataset)
    * ./results/results_[model_id].csv  (test dataset)
* python blender.py
    * Output file: ./final_results/final_results.csv for submissions

More information comming soon!
