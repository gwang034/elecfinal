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
    data = cleaner(train_path, feature_path, morph_path)
    ## Split data into training, validation, query, and testing
    X = data.drop(columns='connected')
    y = data['connected']
    X_train, X_oth, y_train, y_oth = model_selection.train_test_split(X, y, test_size=0.5, random_state=42)
    X_val, X_query, y_val, y_query = model_selection.train_test_split(X_oth, y_oth, test_size=0.25, random_state=42)
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
        ros = RandomOverSampler(random_state=0)
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
    """
    scoring = {"balanced accuracy": make_scorer(balanced_accuracy_score)}
    cv = GridSearchCV(model, 
                      param_grid, 
                      n_jobs = -1, 
                      scoring = "balanced_accuracy",
                      refit = 'balanced_accuracy')
    
    #OverSampling
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(
        valid_X, valid_y
    )

    cv.fit(X_resampled,y_resampled)

    optimum_params = cv.best_params_

    return model.set_params(**optimum_params)
