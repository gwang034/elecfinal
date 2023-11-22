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


def validation(models, X_train, y_train, X_val, y_val, param_grids):
    """
    Function that outputs models with optimal hyperparameters and
    accuracies based on a validation set

    Inputs:
    models: provided model dictionary
    {"RFC": RandomForestClassifier(), 
    "LDA": LinearDiscriminantAnalysis()}
    param_grids: dictionary of combinations of parameters
    e.g. 
    param_grids = {
            "RFC": [{'max_features' : 'sqrt', 'n_jobs' : -1}, 
                    {'max_features' : 'log2', 'n_jobs' : -1}],
            "LDA": [{'solver' : 'lsqr', 'n_jobs' : -1},
                    {'solver' : 'eigen', 'n_jobs' : -1}]
            }
    Outputs: 
    optimum_models: dictionary of provided models with optimum hyperparameters
    accuracies: dictionary of provided models and validation accuracies
    """
    optimum_models = dict()
    accuracies = dict()
    for model in param_grids:
        classifier = models[model]
        prev_acc = 0
        optimum_param = dict()
        for values in param_grids[model]:
            accuracy = []
            #Fitting to the training data with selected hyperparameters
            classifier.set_params(**values)
            classifier.fit(X_train, y_train)

            #Finding the balanced accuracy
            y_hat = classifier.predict(X_val)
            balanced_accuracy = balanced_accuracy_score(y_val, y_hat)
            if balanced_accuracy > prev_acc:
                prev_acc = balanced_accuracy
                optimum_param = values

            accuracy.append(balanced_accuracy)
        accuracies[model] = accuracy
        optimum_models[model] = classifier.set_params(**optimum_param)

    return optimum_models, accuracies

def query_and_predict(tuned_models, X_train, y_train, X_query, y_query):
    """
    Function that outputs the best, trained, model based on a query set as well as 
    accuracies of each inputted model

    Inputs:
    tuned_models: provided tuned models dictionary
    {"RFC": RandomForestClassifier(**best_params), 
    "LDA": LinearDiscriminantAnalysis(**best_params)}

    Outputs: 
    optimum_model: best model
    """ 
    accuracies = dict()
    prev_acc = 0
    for model in tuned_models:
        classifier = tuned_models[model]
        classifier.fit(X_train, y_train)
        yhat = classifier.predict(X_query)
        balanced_accuracy = balanced_accuracy_score(y_query, yhat)
        accuracies[model] = balanced_accuracy
        if balanced_accuracy > prev_acc:
            prev_acc = balanced_accuracy
            optimum_model = classifier
    return optimum_model, accuracies
        



