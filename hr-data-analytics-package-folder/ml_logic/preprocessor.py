# Basic libraries
import numpy as np
import pandas as pd


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder

def preprocess_features_leaving():
    """
    - Create a preprocessing transformer with :
        - 2 Function transformers
        - SimpleImputer
        - MinMaxScaler
        - RobustScaler
        - OneHotEncoder
        - OrdinalEncoder
    """
    #Functions used in basic imputations
    def imputer_critical(x):
        x = x.apply(lambda y: y.map({1 : 1, np.nan: 0}))
        return x

    def imputer_gender(x):
        x = x.apply(lambda y: y.map({'F': 1, 'M': 0}))
        return x

    # Preprocessor
    simp_impute_scale_cols = ['emp_sat_onprem_5']
    robust_scale_cols = ['time_spend_company']
    ohe_scale_cols = ['department', 'geo', 'role']
    ordinal_scale_cols = ['salary']
    minmax_scale_cols = ['last_evaluation','number_project',
                         'average_montly_hours','linkedin_hits',
                         'sensor_stepcount','sensor_heartbeat']

    preproc_leaving = make_column_transformer(
        # Basic imputations
        (FunctionTransformer(imputer_gender,feature_names_out ='one-to-one'), ['gender']),
        (FunctionTransformer(imputer_critical,feature_names_out ='one-to-one'), ['critical']),
        (SimpleImputer(strategy='most_frequent'), simp_impute_scale_cols),

        #Numerical preproc
        (MinMaxScaler(), minmax_scale_cols),
        (RobustScaler(), robust_scale_cols),

        #Categorical preproc
        (OneHotEncoder(sparse_output = False), ohe_scale_cols),
        (OrdinalEncoder(), ordinal_scale_cols),

        #Remaining columns pass
        remainder='passthrough',
        force_int_remainder_cols=False
    )
    return preproc_leaving

def preprocess_features_hiring():
    """
    - Create a preprocessing transformer (for hiring purpose) with :
        - 2 Function transformers
        - MinMaxScaler
        - OneHotEncoder
        - OrdinalEncoder
    """
    #Functions used in basic imputations
    def imputer_critical(x):
        x = x.apply(lambda y: y.map({1 : 1, np.nan: 0}))
        return x

    def imputer_gender(x):
        x = x.apply(lambda y: y.map({'F': 1, 'M': 0}))
        return x

    # Preprocessor
    ohe_scale_cols = ['department', 'geo', 'role']
    minmax_scale_cols = ['average_montly_hours']
    ordinal_scale_cols = ['salary']

    preproc_hiring = make_column_transformer(
        # Basic imputations
        (FunctionTransformer(imputer_gender,feature_names_out ='one-to-one'), ['gender']),
        (FunctionTransformer(imputer_critical,feature_names_out ='one-to-one'), ['critical']),

        #Numerical preproc
        (MinMaxScaler(), minmax_scale_cols),

        #Categorical preproc
        (OneHotEncoder(sparse_output = False), ohe_scale_cols),
        (OrdinalEncoder(), ordinal_scale_cols),

        #Remaining columns pass
        remainder='passthrough',
        force_int_remainder_cols=False
    )
    return preproc_hiring
