
# Basic libraries
import pandas as pd

#Pipelines and transformers
from sklearn.pipeline import make_pipeline

# Models
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Import from .py files
from hr_data_analytics_package_folder.ml_logic.data import clean_data_hiring, \
                                                           clean_data_leaving
from hr_data_analytics_package_folder.ml_logic.preprocessor import preprocess_features_hiring, \
                                                                   preprocess_features_leaving
from hr_data_analytics_package_folder.params import *

def predict_hiring(X_pred):
    """
    This function takes a dataframe for prediction and returns a dictionnary.
    - Retrieve the original data
    - Clean and preprocess data
    - Train model
    - Perform prediction
    - Return ranking of applicants based on
      probability of staying in the company
    """

    print(X_pred)
    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    # Keep info to display output
    display_output = X_pred[["ID","Name"]]

    # Clean data using data.py
    data = clean_data_hiring(data)
    X_pred = clean_data_hiring(X_pred)

    # Create X and y
    X = data.drop(columns=['left_company'])
    y = data['left_company']

    # Drop target from X_pred
    X_pred.drop(columns=['left_company'], inplace=True)
    X_pred_index = X_pred.index

    # Create (X_train_encoded, X_test_encoded) using `preprocessor.py`
    preproc = preprocess_features_hiring()

    # Fitting the pipeline
    model = GradientBoostingClassifier()
    pipeline = make_pipeline(preproc, model)
    pipeline.fit(X, y)

     # Compute prediction
    y_pred = pipeline.predict_proba(X_pred)
    prediction = pd.DataFrame(y_pred,
                              columns = ['prob_stay', 'prob_leave'],
                              index=X_pred_index)

    # Merging input features with prediction for visualization
    X_test_final = pd.merge(display_output,
                            prediction,
                            left_index=True,
                            right_index=True)

    X_test_final.drop(columns=['prob_leave'], inplace=True)
    X_test_final.sort_values('prob_stay', ascending=False, inplace=True)
    X_test_final['prob_stay_percent'] = round(X_test_final['prob_stay'])*100
    X_test_final.drop(columns=['prob_stay'], inplace=True)

    print(f"Ranking  : \n {X_test_final}")
    return X_test_final.to_dict()

def predict_leaving(X_pred):
    """
    This function takes a dataframe for prediction and returns a dictionnary.
    - Retrieve the original data
    - Clean and preprocess data
    - Train model
    - Perform prediction
    - Return a ranking based on the probability
      of a employee leaving company
    """

    print(X_pred)
    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    # Keep info to display output
    display_output = X_pred[["ID","Name"]]

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
    model = XGBClassifier()
    pipeline = make_pipeline(preproc, model)
    pipeline.fit(X, y)

    # Compute prediction
    y_pred = pipeline.predict_proba(X_pred)
    # breakpoint()
    prediction = pd.DataFrame(y_pred,
                              columns = ['prob_stay', 'prob_leave'],
                              index=X_pred_index)

    # Merging input features with prediction for visualization
    X_test_final = pd.merge(display_output,prediction,
                            left_index=True,right_index=True)
    X_test_final.drop(columns=['prob_stay'], inplace=True)
    X_test_final.sort_values('prob_leave', ascending=False, inplace=True)
    X_test_final['prob_leave_percent'] = round(X_test_final['prob_leave'])*100
    X_test_final.drop(columns=['prob_leave'], inplace=True)

    print(f"Ranking  : \n {X_test_final}")
    return X_test_final.to_dict()
