# Basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os.path import exists
from hr_data_analytics_package_folder.params import *
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, \
                                    RobustScaler, \
                                    MinMaxScaler, \
                                    OneHotEncoder

def plot_time_spend_company():
    #Delete previous any previous png file
    file_path = 'raw_data/time_spend_company.png'
    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    # feature_variable = "time_spend_company"
    n_bins=8

    ax = (data
        .loc[lambda x: x["left_Company"] == 0]
        ["time_spend_company"]
        .plot
        .hist(label="Did Not Leave Company",density=True,bins=n_bins,alpha=0.3)
        )

    (data
    .loc[lambda x: x["left_Company"] == 1]
    ["time_spend_company"]
    .plot
    .hist(ax=ax, label="Left Company",density=True,bins=n_bins,alpha=0.3)
    )

    plt.xlabel("time_spend_company")
    plt.legend()
    plt.savefig('raw_data/time_spend_company.png')
    plt.close()
    return 'raw_data/time_spend_company.png'


def plot_salary():
    #Delete previous any previous png file
    file_path = 'raw_data/salary.png'
    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    salary_ct = pd.crosstab(data['salary'], data['left_Company'], normalize='index') * 100

    ax = salary_ct.plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(10, 6))
    plt.title('Employee Retention by Salary Level (Percentage)')
    plt.xlabel('Salary Level')
    plt.ylabel('Percentage')
    plt.legend(title='Left Company', labels=['Stayed', 'Left'])

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center')

    plt.savefig('raw_data/salary.png')
    plt.close()
    return 'raw_data/salary.png'


def plot_feature_importance():
    #Delete previous any previous png file
    file_path = 'raw_data/feature_importance.png'
    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    data.drop(columns = ['ID','Name','Rising_Star', 'Trending Perf', 'Talent_Level',
                     'Validated_Talent_Level', 'EMP_Sat_OnPrem_1', 'EMP_Sat_OnPrem_2',
                     'EMP_Sat_OnPrem_3','EMP_Sat_Remote_3', 'EMP_Sat_Remote_4','EMP_Sat_Remote_5',
                     'EMP_Engagement_2','EMP_Engagement_3','EMP_Engagement_4',
                     'EMP_Engagement_5','CSR Factor','sales'], inplace = True)

    data = data.rename(columns={
        'Sensor_Heartbeat(Average/Min)': 'Sensor_Heartbeat',
        'Sensor_Proximity(1-highest/10-lowest)': 'Sensor_Proximity'
    })

    # Convert column names to lower snake case
    data.columns = data.columns.str.lower()\
        .str.replace(' ', '_')\
            .str.replace('-', '_')\
                .str.replace('.', '_')

    # combine all the men_leave and the women_leave column
    data['leave'] = data['women_leave'].fillna(data['men_leave'])
    data['leave'] = data['leave'].fillna(0)
    data.drop(columns = ['women_leave', 'men_leave'], inplace = True)

    # impute binary missing values for critical column
    data.loc[data['critical'].isna(), 'critical'] = 0

    # covert gender column into binary classification column
    data['gender'] = data['gender'].map({'F': 1, 'M': 0})

    # impute missing values with mode as these are ranking/ordinal columns
    columns_to_impute = ['emp_sat_onprem_4', 'emp_sat_onprem_5']
    imputer = SimpleImputer(strategy='most_frequent')
    data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

    # time_spend_company is the only numeric variable with outliers
    # therefore we will use the Robust Scaler.
    rb_scaler = RobustScaler()
    data['time_spend_company'] = rb_scaler\
        .fit_transform(data[['time_spend_company']])


    # for all other numerical variables, use the minmaxscaler
    minmaxscaler_columns = ['last_evaluation', 'number_project', 'average_montly_hours',
                        'linkedin_hits', 'sensor_stepcount', 'sensor_heartbeat']

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data using MinMaxScaler
    data[minmaxscaler_columns] = scaler.fit_transform(data[minmaxscaler_columns])

    # ordinal encoding for salary
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(data[['salary']])
    data['encoded_salary'] = ordinal_encoder\
        .transform(data[['salary']]) # 1 = low, 2 = medium, 3 = high
    data.drop(columns = ['salary'], inplace = True)

    # OneHotEncoding for department, geo and role
    ohe_columns = ['department', 'geo', 'role']
    ohe = OneHotEncoder(sparse_output = False)
    ohe_data = ohe.fit_transform(data[ohe_columns])
    ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(ohe_columns))
    encoded_data = pd.concat([data, ohe_df], axis=1)
    encoded_data.drop(columns = ['department', 'geo', 'role'], inplace = True)

    # define x and y variables
    X = encoded_data.drop(columns = 'left_company')
    y = encoded_data['left_company']

    # remove highly correlated features >= 0.7 or <= -0.7
    X.drop(columns = ['emp_sat_onprem_4', 'percent_remote',
                      'emp_sat_remote_2', 'emp_sat_remote_1',
                      'emp_engagement_1'], inplace = True)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    feature_names = X.columns.tolist()

    rf_feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    })

    # Filter features with importance greater than 0.01
    rf_feature_importance = rf_feature_importance[rf_feature_importance['Importance'] > 0.01]

    # Plotting feature importance using Seaborn for better visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=rf_feature_importance.\
        sort_values(by='Importance', ascending=False), palette='viridis')
    plt.title('Top Important Features from Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('raw_data/feature_importance.png')
    plt.close()
    return 'raw_data/feature_importance.png'


def plot_number_projects():
    #Delete previous any previous png file
    file_path = 'raw_data/number_projects.png'
    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    plt.figure(figsize=(12, 6))
    sns.countplot(x='number_project',
                  hue='left_Company',
                  data=data,
                  palette={0: 'green', 1: 'red'})
    plt.title('Distribution of Number of Projects by Employee Status')
    plt.xlabel('Number of Projects')
    plt.ylabel('Count of Employees')
    plt.legend(title='Left Company', labels=['Stayed', 'Left'])
    plt.savefig('raw_data/number_projects.png')
    plt.close()
    return 'raw_data/number_projects.png'


def plot_step_count():
    #Delete previous any previous png file
    file_path = 'raw_data/step_count.png'
    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    pivot_table = data.pivot_table(values='Sensor_StepCount',
                                   index='Percent_Remote',
                                   columns='left_Company',
                                   aggfunc='mean')

    pivot_table.plot(kind='bar', figsize=(12, 6), color=['green', 'red'])
    plt.title('Stepcount by percent remote and Employee Status')
    plt.xlabel('Percent remote')
    plt.ylabel('Stepcount')
    plt.legend(title='Left Company', labels=['Stayed', 'Left'], loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('raw_data/step_count.png')
    plt.close()
    return 'raw_data/step_count.png'
