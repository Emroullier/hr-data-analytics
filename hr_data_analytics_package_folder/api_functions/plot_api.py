# Basic libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from hr_data_analytics_package_folder.params import *

def plot_violin(y : str):
    """
    This function takes a str as input and returns
    the corresponding violin plot
    """
    y_norm = y
    if y in ['Sensor_Heartbeat(Average/Min)',
             'Sensor_Proximity(1-highest/10-lowest)']:
        if y == 'Sensor_Heartbeat(Average/Min)':
            y_norm = 'Sensor_Heartbeat'
        elif y == 'Sensor_Proximity(1-highest/10-lowest)':
            y_norm = 'Sensor_Proximity'

    #Delete previous any previous png file
    file_path = f'raw_data/violin_{y_norm}.png'

    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    # Other inputs
    x='left_Company'
    hue='left_Company'

    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    plt.figure(figsize=(10, 6))
    sns.violinplot(x = x,
                   y = y,
                   hue = hue,
                   data=data,
                   palette={0: 'green', 1: 'red'},
                   dodge=False,
                   legend=False)
    plt.title(f'Distribution of {y_norm} by Employee Status')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks([0, 1], ['Stayed', 'Left'])
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(f'raw_data/violin_{y_norm}.png')
    plt.close()
    return f'raw_data/violin_{y_norm}.png'

def cross_tab_count_feat(x:str):
    """
    This function takes a str as input and returns
    the corresponding crosstab plot locally
    """
    y = 'left_Company'

    #Delete previous any previous png file
    file_path = f'raw_data/crosstab_count_feat_{x}.png'

    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")


    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    labels = ['No', 'Yes']
    gender_ct = pd.crosstab(data[x], data[y])
    gender_ct.plot(kind='bar')
    plt.title(f'{x} Distribution')
    plt.xlabel(x)
    plt.ylabel('Count')
    plt.legend(title='Left Company', labels=labels)
    plt.tight_layout()
    plt.savefig(f'raw_data/crosstab_count_feat_{x}.png')
    plt.close()
    return f'raw_data/crosstab_count_feat_{x}.png'

def freq_feat(input : str):
    """
    This function takes a str as input and returns
    the corresponding frequence vs feature plot locally
    """

    #Delete previous any previous png file
    file_path = f'raw_data/freq_feat_{input}.png'

    file_exists = exists(file_path)
    if file_exists:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")

    #Retrieve dataset from local directory
    data = pd.read_csv(DATA_HR)

    n_bins=8
    ax = (data
          .loc[lambda x: x["left_Company"] == 0]
          [input]
          .plot
          .hist(label="Did Not Leave Company",density=True,bins=n_bins,alpha=0.3)
         )

    (data
     .loc[lambda x: x["left_Company"] == 1]
     [input]
     .plot
     .hist(ax=ax, label="Left Company",density=True,bins=n_bins,alpha=0.3)
    )
    plt.xlabel(input)
    plt.legend()
    plt.savefig(f'raw_data/freq_feat_{input}.png')
    plt.close()
    return f'raw_data/freq_feat_{input}.png'
