# Basic libraries
import pandas as pd
import numpy as np
from colorama import Fore, Style

#Pipelines and transformers
from sklearn.pipeline import make_pipeline

# Models
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Import from .py files
from ml_logic.data import clean_data_leaving, clean_data_hiring
from ml_logic.preprocessor import preprocess_features_hiring,\
                                  preprocess_features_leaving

from plots.plot_func import plot_time_spend_company,\
                            plot_feature_importance, \
                            plot_salary,\
                            plot_number_projects,\
                            plot_step_count

from params import *

def predict_leaving(X_pred, model):
    """
    - Retrieve the original data
    - Clean and preprocess data
    - Train model
    - Perform prediction
    - Return a ranking based on the probability
      of a employee leaving company
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: prediction of employee leaving (output)" + Style.RESET_ALL)
    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    # Clean data using data.py
    data = clean_data_leaving(data)
    X_pred = clean_data_leaving(X_pred)
    print("✅ data cleaned")

    # Create X and y
    X = data.drop(columns=['left_company'])
    y = data['left_company']

    # Drop target from X_pred
    X_pred.drop(columns=['left_company'], inplace=True)
    X_pred_index = X_pred.index

    # Create (X_train_encoded, X_test_encoded) using `preprocessor.py`
    preproc = preprocess_features_leaving()

    # Fitting the pipeline
    pipeline = make_pipeline(preproc, model)
    pipeline.fit(X, y)

    # Compute prediction
    y_pred = pipeline.predict_proba(X_pred)
    prediction = pd.DataFrame(y_pred,
                              columns = ['prob_stay', 'prob_leave'],
                              index=X_pred_index)

    # Merging input features with prediction for visualization
    X_test_final = pd.merge(X_pred, prediction,
                            left_index=True, right_index=True)
    X_test_final.drop(columns=['prob_stay'], inplace=True)
    X_test_final.sort_values('prob_leave', ascending=False, inplace=True)

    print(f"Ranking  : \n {X_test_final}")
    print(f"✅ pred_leaving() done")

def predict_hiring(X_pred, model):
    """
    - Retrieve the original data
    - Clean and preprocess data
    - Train model
    - Perform prediction
    - Return ranking of applicants based on
      probability of staying in the company
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: prediction of hiring an applicant" + Style.RESET_ALL)
    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    # Clean data using data.py
    data = clean_data_hiring(data)
    X_pred = clean_data_hiring(X_pred)
    print("✅ data cleaned")

    # Create X and y
    X = data.drop(columns=['left_company'])
    y = data['left_company']

    # Drop target from X_pred
    X_pred.drop(columns=['left_company'], inplace=True)
    X_pred_index = X_pred.index

    # Create (X_train_encoded, X_test_encoded) using `preprocessor.py`
    preproc = preprocess_features_hiring()

    # Fitting the pipeline
    pipeline = make_pipeline(preproc, model)
    pipeline.fit(X, y)

     # Compute prediction
    y_pred = pipeline.predict_proba(X_pred)
    prediction = pd.DataFrame(y_pred,
                              columns = ['prob_stay', 'prob_leave'],
                              index=X_pred_index)

    # Merging input features with prediction for visualization
    X_test_final = pd.merge(X_pred, prediction, left_index=True, right_index=True)
    X_test_final.drop(columns=['prob_leave'], inplace=True)
    X_test_final.sort_values('prob_stay', ascending=False, inplace=True)

    print(f"Ranking  : \n {X_test_final}")
    print(f"✅ predict_hiring() done")


def generate_input_leaving(num):
    """
    - Generate dummy input features ONLY for testing application
    - Beware : Obvious data leakage.
    - This will replaced by user input. Pandas Dataframe expected.
    """
    print(Fore.MAGENTA + f"\n ⭐️ Input data : {num} random employees considered here" + Style.RESET_ALL)
    data = pd.read_csv(DATA_HR)
    input_features = data.sample(num)
    print(input_features)
    return input_features

def generate_input_hiring(num):
    """
    - Generate dummy input features ONLY for testing application
    - Beware : Obvious data leakage.
    - This will replaced by user input. Pandas Dataframe expected.
    """
    print(Fore.MAGENTA + f"\n ⭐️ Input data : {num} applicants considered here" + Style.RESET_ALL)
    data = pd.read_csv(DATA_HR)
    # Check the applicants for a specific job
    query = f"@data['Department'] == 'Sales'      and\
            @data['GEO']=='UK'                    and\
            @data['average_montly_hours'] > 200   and\
            @data['salary']=='low'                 and\
            @data['Role']=='Level 2-4'"

    input_features = data.query(query)
    input_features = input_features.sample(num)
    print(input_features)
    return input_features

def say_hello():
    print('Hello World !')

if __name__ == '__main__':
    try:
        # *** predict_leaving : Lists employees likely to quit the company ***
        # predict_leaving(generate_input_leaving(10),XGBClassifier())

        # *** predict_hiring : Ranks a set of applicants applying for the same job.***
        # predict_hiring(generate_input_hiring(10),GradientBoostingClassifier())

        # Plotting
        plot_time_spend_company()
        # plot_feature_importance()
        # plot_salary()
        # plot_number_projects()
        # plot_step_count()

        # say_hello()

    except:
        import sys
        import traceback
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
