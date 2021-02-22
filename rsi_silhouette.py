# %%
import pandas as pd
import numpy as np

import pickle
from time import time

from tqdm.auto import tqdm
from sys import stdout

from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score, f1_score, adjusted_mutual_info_score, calinski_harabasz_score, silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import seaborn as sns

def importdata(file_name):
    print('Importing the data...')
    st = time()
    with open(file_name,'rb') as data:
        df = pickle.load(data)
    end = time()
    print('Done importing in '+str(round(end-st,2))+' seconds.')

    return df

def convert_to_dict(data):
    print('Converting data...')
    st = time()
    temp_dict = {}
    for df in data:
        temp_dict[int(df['Subject'].iloc[0][4:])] = df.copy()
        temp_dict.get(int(df['Subject'].iloc[0][4:])).reset_index(drop=True, inplace=True)
    end = time()
    print('Done converting in '+str(round(end-st,2))+' seconds.')

    return temp_dict

def get_results(df):
    temp = df.copy()
    test_df = (
            ('phase1_vs_phase3',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase3')]),
            ('phase1_vs_phase5',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase5')]),
            ('phase3_vs_phase5',temp[(temp['Phase'] == 'phase3') | (temp['Phase'] == 'phase5')]),
            ('phase1_vs_phase2',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase2')]),
            ('phase1_vs_phase4',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase4')]),
            ('phase3_vs_phase2',temp[(temp['Phase'] == 'phase3') | (temp['Phase'] == 'phase2')]),
            ('phase3_vs_phase4',temp[(temp['Phase'] == 'phase3') | (temp['Phase'] == 'phase4')]),
            ('phase5_vs_phase2',temp[(temp['Phase'] == 'phase5') | (temp['Phase'] == 'phase2')]),
            ('phase5_vs_phase4',temp[(temp['Phase'] == 'phase5') | (temp['Phase'] == 'phase4')]) 
        )
    result = {}
    with tqdm(total=len(test_df), file=stdout) as pbar:
        for data_id, dataset in test_df:
            pbar.set_description('    Dataset')
            result[data_id] = get_metric(dataset)
            pbar.update(1)
    
    return result

def get_metric(dataset):
    #print('Training the model...')
    # Initialize the variables
    scores = []
    # Split x and y as we know the labels
    X = dataset.drop(['Time','Subject','Phase'], axis=1).values
    y_ = dataset['Phase']
    # label binarize the target class
    classes = y_.unique()
    y = label_binarize(y_, classes=classes)
    # the bigger the number the more separated they are. Unkown ground truth
    scores.append(calinski_harabasz_score(X, y))
    # silhouette coefficient
    scores.append(silhouette_score(X, y, metric='mahalanobis'))
    # Davies-Bouldin score. The less the better
    scores.append(davies_bouldin_score(X, y))

    return scores

def execute_all(subjects_dict):
    result = {}
    sub = 1
    #subjects_to_test = subjects_dict.keys():
    subjects_to_test = range(455,462)
    for subject in subjects_to_test:
        print('Subject: {} Progress:({}/{})'.format(subject,sub,len(subjects_to_test)))
        result[subject] = get_results(subjects_dict.get(subject))
        sub += 1

    return result

def results_to_dataframe(results_):
        temp = {(level1_key, level2_key): values
                for level1_key, level2_dict in results_.items()
                for level2_key, values in level2_dict.items()}
        temp_df = pd.DataFrame(temp).T
        temp_df.columns = ['CalinskiHarabasz','Silhouette_Mahanlanobis','DaviesBouldin']
        temp_df = temp_df.reset_index().rename(columns={'level_0':'Subject','level_1':'Phase_vs_phase','level_2':'model'})
        temp_df.to_csv('silhouette_mahalanobis_results_new.csv', index=False)

        return None
# %%
data = importdata('subjects_151_new.data')
df = convert_to_dict(data)
# %%
result = execute_all(df)
# %%
results_to_dataframe(result)
# %%
def import_results(file_loc):
    return pd.read_csv(file_loc)

def subset_data(data):
    return data[(data['Phase_vs_phase'].str.contains('phase1_'))].sort_values(['Subject','Phase_vs_phase'])

def standard_data(data):
    df = data[['Subject','Phase_vs_phase','Silhouette_Mahanlanobis']].copy()
    df = df.pivot(index='Subject',columns='Phase_vs_phase',values='Silhouette_Mahanlanobis').reset_index()
    df = df.loc[:,df.columns != 'Phase_vs_phase']
    df.columns = ['Subject','Ph1-Ph2','Ph1-Ph3','Ph1-Ph4','Ph1-Ph5']

    return  df.set_index('Subject',drop=True)

def convert_df_to_dict(df):
    return {key: values.groupby('Phase_vs_phase')['Silhouette_Mahanlanobis'].apply(float).to_dict() for key,values in df.groupby('Subject')}

def calculate_index_modified(distances):
    distances['Max_Delta_Stress'] = np.nan
    distances['Recovery_Delta_Last_Phase'] = np.nan
    distances['Resilience_Index'] = np.nan
    # Calculate of maximum point of stress before the last stressor phase and get the recovery phase delta
    for element in distances.index.values:
        #if pd.isnull(distances['Ph1-Ph6'].loc[element]) == True:
        if pd.isnull(distances['Ph1-Ph5'].loc[element]) == True:
            if pd.isnull(distances['Ph1-Ph4'].loc[element]) == True:
                distances['Max_Delta_Stress'].loc[element] = distances['Ph1-Ph2'].loc[element]
                distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph3'].loc[element]
        else:
            distances['Max_Delta_Stress'].loc[element] = distances[['Ph1-Ph2','Ph1-Ph4']].loc[element].max()
            distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph5'].loc[element]
        #else:
        #    distances['Max_Delta_Stress'].loc[element] = distances[['Ph1-Ph2','Ph1-Ph4']].loc[element].max()
        #    distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph6'].loc[element]
    # Get the maximum value of the population of delta stress
    distances['population_max_delta_stress'] = distances['Max_Delta_Stress']-distances['Recovery_Delta_Last_Phase']
    distances['stress_factor'] = (distances['Max_Delta_Stress']-distances['Max_Delta_Stress'].min())/(distances['Max_Delta_Stress'].max()-distances['Max_Delta_Stress'].min())
    # Get the resilience index, the more positive the better resilience to stress.
    distances['Resilience_Index'] = (distances['Max_Delta_Stress']-distances['Recovery_Delta_Last_Phase'])/distances['population_max_delta_stress'].max()

    sns.distplot(distances['Resilience_Index'], hist=True, kde=True, color='darkblue',hist_kws={'edgecolor':'black'})
    plt.ylabel('Density')
    plt.xlabel('Resilience Index')
    plt.show()

    sns.boxplot(x='Resilience_Index', orient='h',data=distances)
    plt.show()

    return distances

# %%
data = import_results('Results/silhouette_mahalanobis_results_complete.csv')
results = subset_data(data)
# %%
distances = standard_data(results)
# %%
dist_sil = calculate_index_modified(distances)
# %%
dist_sil.to_csv('silhouette_distances.csv')
# %%
