
import pandas as pd

def clean_data_leaving(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Clean raw data by :
        -  Removing unwanted columns
        -  Renaming columns
        -  Converting column names to lower snake case
        -  Combining all the men_leave and the women_leave column
        -  Removing highly correlated features >= 0.7 or <= -0.7
    """
    # Remove unwanted columns
    dropped_columns = ['ID','Name','Rising_Star', 'Trending Perf',
                       'Talent_Level','Validated_Talent_Level',
                       'EMP_Sat_OnPrem_1', 'EMP_Sat_OnPrem_2',
                        'EMP_Sat_OnPrem_3','EMP_Sat_Remote_3',
                        'EMP_Sat_Remote_4','EMP_Sat_Remote_5',
                        'EMP_Engagement_2','EMP_Engagement_3',
                        'EMP_Engagement_4','EMP_Engagement_5',
                        'CSR Factor','sales']

    df.drop(columns = dropped_columns, inplace = True)

    # Rename columns
    df = df.rename(columns={
                'Sensor_Heartbeat(Average/Min)': 'Sensor_Heartbeat',
                'Sensor_Proximity(1-highest/10-lowest)': 'Sensor_Proximity'
            })

    # Convert column names to lower snake case
    df.columns = df.columns.str.lower()\
                            .str.replace(' ', '_')\
                            .str.replace('-', '_')\
                            .str.replace('.', '_')

    # combine all the men_leave and the women_leave column
    df['leave'] = df['women_leave'].fillna(df['men_leave'])
    df['leave'] = df['leave'].fillna(0)
    df.drop(columns = ['women_leave', 'men_leave'], inplace = True)

    # remove highly correlated features >= 0.7 or <= -0.7
    df.drop(columns=['emp_sat_onprem_4','percent_remote',
                    'emp_sat_remote_2','emp_sat_remote_1',
                    'emp_engagement_1'], inplace = True)

    #Dev purpose
    df.drop(columns=['critical'])

    # print("✅ data cleaned")
    return df

def clean_data_hiring(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Clean raw data by :
        -  Removing unwanted columns
        -  Renaming columns
        -  Converting column names to lower snake case
        -  Combining all the men_leave and the women_leave column
        -  Removing highly correlated features >= 0.7 or <= -0.7
        -  Performing applicant specific preparations
    """
    # Remove unwanted columns
    dropped_columns = ['ID','Name','Rising_Star', 'Trending Perf',
                       'Talent_Level','Validated_Talent_Level',
                       'EMP_Sat_OnPrem_1', 'EMP_Sat_OnPrem_2',
                        'EMP_Sat_OnPrem_3','EMP_Sat_Remote_3',
                        'EMP_Sat_Remote_4','EMP_Sat_Remote_5',
                        'EMP_Engagement_2','EMP_Engagement_3',
                        'EMP_Engagement_4','EMP_Engagement_5',
                        'CSR Factor','sales']

    df.drop(columns = dropped_columns, inplace = True)

    # Rename columns
    df = df.rename(columns={
                'Sensor_Heartbeat(Average/Min)': 'Sensor_Heartbeat',
                'Sensor_Proximity(1-highest/10-lowest)': 'Sensor_Proximity'
            })

    # Convert column names to lower snake case
    df.columns = df.columns.str.lower()\
                            .str.replace(' ', '_')\
                            .str.replace('-', '_')\
                            .str.replace('.', '_')

    # combine all the men_leave and the women_leave column
    df['leave'] = df['women_leave'].fillna(df['men_leave'])
    df['leave'] = df['leave'].fillna(0)
    df.drop(columns = ['women_leave', 'men_leave'], inplace = True)

    # remove highly correlated features >= 0.7 or <= -0.7
    df.drop(columns=['emp_sat_onprem_4','percent_remote',
                    'emp_sat_remote_2','emp_sat_remote_1',
                    'emp_engagement_1'], inplace = True)

    # Specific preparation for applicants
    # job_posting = ['department','geo','role',
    #                'average_montly_hours',
    #                'salary','critical']

    job_posting = ['department','geo','role',
                'average_montly_hours',
                'salary']

    applicant = ['will_relocate', 'gender']
    df = df[job_posting + applicant + ['left_company']]

    return df
