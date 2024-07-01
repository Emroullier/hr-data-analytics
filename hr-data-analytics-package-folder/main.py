# Basic libraries
import pandas as pd
from colorama import Fore, Style

#Pipelines and transformers
from sklearn.pipeline import make_pipeline

# Models
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Import from .py files
from ml_logic.data import clean_data_leaving, clean_data_hiring
from ml_logic.preprocessor import preprocess_features_hiring, preprocess_features_leaving
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
    print(Fore.MAGENTA + "\n ⭐️ Use case: prediction of employee leaving" + Style.RESET_ALL)
    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    # Clean data using data.py
    # Clean database
    data = clean_data_leaving(data)
    # Clean data used for prediction
    X_pred = clean_data_leaving(X_pred)

    # Create X and y
    X = data.drop(columns=['left_company'])
    y = data['left_company']

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

    # Create X and y
    X = data.drop(columns=['left_company'])
    y = data['left_company']

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

def generate_input_leaving():
    """
    - Generate dummy input features ONLY for testing application
    - Beware : Obvious data leakage.
    - This will replaced by user input. Pandas Dataframe expected.
    """
    data = pd.read_csv(DATA_HR)
    input_features = data.iloc[470:485]
    print(type(input_features))
    return input_features

def generate_input_hiring():
    """
    - Generate dummy input features ONLY for testing application
    - Beware : Obvious data leakage.
    - This will replaced by user input. Pandas Dataframe expected.
    """
    data = pd.read_csv(DATA_HR)
    data = clean_data_hiring(data)
    data.drop(columns=['left_company'], inplace = True)

    # Check the applicants for a specific job
    query = "@data['department'] == 'Operations'      and\
            @data['geo']=='UK'                    and\
            @data['role']== 'Level 2-4'           and\
            @data['average_montly_hours'] > 200   and\
            @data['salary']=='low'"
    input_features = data.query(query)
    return input_features

def say_hello():
    print('Hello World !')

if __name__ == '__main__':
    try:
        # *** predict_leaving : Lists employees likely to quit the company ***
        predict_leaving(generate_input_leaving(),XGBClassifier())

        # *** predict_hiring : Ranks a set of applicants applying for the same job.***
        # predict_hiring(generate_input_hiring(),GradientBoostingClassifier())

        # say_hello()
    except:
        import sys
        import traceback
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
