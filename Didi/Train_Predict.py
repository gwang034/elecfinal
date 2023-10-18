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

## REMOVE train/test/val/query split because it's in grace's code
def train_predict_submission(train, query, models):
    """
    Function that takes in a dataframe of data and outputs 
    a fitted "optimal" model

    Inputs:
    - train: training set
    - valid: validation set 
    - query: query set
    - models: list of models to train and predict on, with set 
    parameters already.

    Outputs:
    ##TO DO: put in best_clf oputput
    - best_clf: The optimum classifier fitted over training data

    - accuracy_score: list of accuracies based on order of models
    passed.
    """
    accuracy_score = []
    for model in models:

        #Defining a pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        X = query.drop('connected', axis=1)
        y = query["connected"]

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
        accuracy_score.append(balanced_accuracy)

    return accuracy_score


def validation(model, valid, param_grid):
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
    valid: validation set of data (pandas df)

    Outputs: 
    clf: provided model with optimum hyperparameters
    """
    scoring = {"balanced accuracy": make_scorer(balanced_accuracy_score)}
    cv = GridSearchCV(model, 
                      param_grid, 
                      n_jobs = -1, 
                      scoring = "balanced_accuracy",
                      refit = 'balanced_accuracy')
    
    cv.fit(valid.drop('connected', axis=1),valid["connected"])

    optimum_params = cv.best_params_

    return model.set_params(optimum_params)





