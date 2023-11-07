import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer
from imblearn.over_sampling import RandomOverSampler
import sys
import numpy as np
from sklearn import model_selection
from Data.data_cleaner import cleaner
import random

### CHANGE PATH FOR YOUR COMPUTER
GITHUB_PATH = '/Users/gracewang/Documents/GitHub/elecfinal'
sys.path.insert(0, GITHUB_PATH)

def clean_split(train_path, feature_path=None, morph_path=None):
    """
    Function that performs data cleaning and splitting

    Inputs:
    - train_path: the path to the training data
    - feature_path: the path to the feature data (default None)
    - morph_path: the path to the morph data (default None)

    Outputs:
    - X_train, y_train: training data set
    - X_val, y_val: validation data set
    - X_query, y_query: query data set
    """
    # clean the data and perform feature engineering
    data = cleaner(train_path, feature_path, morph_path)

    # perform stratified sampling of pre-synaptic neurons
    pre_nucleus_ids = pd.unique(data["pre_nucleus_id"])
    #print(len(pre_nucleus_ids))

    # Use 60% of the pre-nucleus ids and 60% of the post-nucleus ids in the training\
    #print(len(pre_nucleus_ids))
    train_nucleus_idx = random.sample(range(0, len(pre_nucleus_ids)), int(np.floor(0.6*len(pre_nucleus_ids))))
    train_nucleus_ids = pre_nucleus_ids[train_nucleus_idx]
    training = data[data["pre_nucleus_id"].isin(train_nucleus_ids)]
    X_train = training.drop(columns='connected')
    y_train = training['connected']
    pre_nucleus_ids = np.delete(pre_nucleus_ids, train_nucleus_idx)

    # Use 20% for query set
    #print(len(pre_nucleus_ids))
    query_nucleus_idx = random.sample(range(0, len(pre_nucleus_ids)), int(np.floor(0.5*len(pre_nucleus_ids))))
    query_nucleus_ids = pre_nucleus_ids[query_nucleus_idx]
    query = data[data["pre_nucleus_id"].isin(query_nucleus_ids)]
    X_query = query.drop(columns='connected')
    y_query = query['connected']
    pre_nucleus_ids = np.delete(pre_nucleus_ids, query_nucleus_idx)


    # Use 20% for validation
    #print(len(pre_nucleus_ids))
    validation = data[data["pre_nucleus_id"].isin(pre_nucleus_ids)]
    X_val = validation.drop(columns='connected')
    y_val = validation['connected']

    # ## Split data into training, validation, query, and testing
    # X = data.drop(columns='connected')
    # y = data['connected']
    # X_train, X_oth, y_train, y_oth = model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
    # X_val, X_query, y_val, y_query = model_selection.train_test_split(X_oth, y_oth, test_size=0.25, random_state=42)


    return X_train, X_val, X_query, y_train, y_val, y_query





def train_n_predict(train_X, train_y, query_X, query_y, models):
    """
    Function that takes in a dataframe of data and outputs 
    a fitted "optimal" model

    Inputs:
    - train: training set
    - query: query set
    - models: dictionary of (model_name : model function) to train and predict on, with optimized 
    parameters already.

    Outputs:
    - best_clf: The optimum classifier function fitted over training data

    - accuracy_score: list of accuracies based on order of models
    passed.
    """
    accuracy_score = {}
    for model in models:

        #Defining a pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", models[model])
        ])

        X = query_X.copy()
        y = query_y.copy()

        #OverSampling
        ros = RandomOverSampler(random_state=0, sampling_strategy = 'minority')
        X_resampled, y_resampled = ros.fit_resample(
            X, y
        )

        # fit model
        pipe.fit(X_resampled, y_resampled)

        #Predicting from soft labels
        ## TO DO: FIX THIS MAYBE
        X["pred"] = pipe.predict_proba(X)[:, 1]
        balanced_accuracy = balanced_accuracy_score(
            y, 
            X["pred"] > 0.5)
        accuracy_score[model] = balanced_accuracy

    best_clf_name = max(accuracy_score, key = accuracy_score.get)
    best_clf = models[best_clf_name]

    ### TRAIN OVER TRAINING DATA

    ros = RandomOverSampler(random_state=0, sampling_strategy = 'minority')
    train_X_resampled, train_y_resampled = ros.fit_resample(
            train_X, train_y
        )

    best_clf.fit(train_X_resampled, train_y_resampled)
    return accuracy_score, best_clf


def validation(model, valid_X, valid_y, param_grid):
    """
    Function that outputs a model with optimal hyperparameters
    based on a validation set using grid search

    Inputs:
    model: provided model
    param_grid: dictionary of parameters and values to validate on
    e.g. 
    {'C': [0.001,0.01,0.1,1,10], 
    'gamma':[0.1,1,10,100], 
    'kernel':('linear', 'rbf')}
    valid_X: validation X of data (pandas df)
    valid_y: validation y of data

    Outputs: 
    clf: provided model with optimum hyperparameters
    perf: performance of the model during CV
    """
    scoring = {"balanced accuracy": make_scorer(balanced_accuracy_score)}
    cv = GridSearchCV(model, 
                      param_grid, 
                      n_jobs = -1, 
                      scoring = "balanced_accuracy",
                      refit = 'balanced_accuracy')
    
    #OverSampling
    ros = RandomOverSampler(random_state=0, sampling_strategy = 'minority')
    X_resampled, y_resampled = ros.fit_resample(
        valid_X, valid_y
    )

    cv.fit(X_resampled,y_resampled)

    optimum_params = cv.best_params_

    return model.set_params(**optimum_params)
