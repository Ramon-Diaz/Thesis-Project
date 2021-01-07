# %%
import pandas as pd
import numpy as np

import pickle
from time import time

from tqdm.auto import tqdm
from sys import stdout

from sklearn.preprocessing import label_binarize

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import roc_auc_score, f1_score

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

def get_results(df, models):
    temp = df.copy()
    test_df = (
                    ('phase1_vs_phase2',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase2')]),
                    ('phase1_vs_phase3',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase3')]),
                    ('phase1_vs_phase4',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase4')]),
                    ('phase1_vs_phase5',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase5')]),
                )
    result = {}
    for model_name, model in models:
        temp = {}
        print('  Model: {}'.format(model_name))
        with tqdm(total=len(test_df), file=stdout) as pbar:
            for data_id,dataset in test_df:
                pbar.set_description('    Dataset')
                temp[data_id] = train_cluster(dataset, model)
                pbar.update(1)
        result[model_name] = temp

    return result

def train_cluster(dataset, model):
    #print('Training the model...')
    st = time()
    # Initialize the variables
    scores = []
    features = ['Electromyography', 'BloodVolume', 'Breathing','SkinConductance', 'CorporalTemperature']
    # Split x and y as we know the labels
    X = dataset[features].values
    y_ = dataset['Phase']
    # label binarize the target class
    classes = y_.unique()
    y = label_binarize(y_, classes=classes)
    # Fit the model and return the labels
    y_pred = model.fit_predict(X)
    # Compute the AUC score
    auc = roc_auc_score(y, y_pred)
    scores.append(1-auc if auc<0.5 else auc)
    #scores.append(auc)
    scores.append(f1_score(y, y_pred))
    end = time()
    #print('Done training in {:.2f} minutes.'.format((end-st)/60))
    #print('AUC {:.4f} and F1-Score: {:.2f}'.format(scores[0],scores[1]))

    return scores

def execute_all(models, subjects_dict):
    result = {}
    sub = 1
    for subject in subjects_dict.keys():
    #for subject in [100,414]:
        print('Subject: {} Progress:({}/{})'.format(subject,sub,len(subjects_dict.keys())))
        result[subject] = get_results(subjects_dict.get(subject), models)
        sub += 1

    return result

def results_to_dataframe(results_):
        temp = {(level1_key, level2_key, level3_key): values
                for level1_key, level2_dict in results_.items()
                for level2_key, level3_dict in level2_dict.items()
                for level3_key, values      in level3_dict.items()}
        temp_df = pd.DataFrame(temp).T
        temp_df.columns = ['AUC','F1']
        temp_df = temp_df.reset_index().rename(columns={'level_0':'Subject','level_1':'Phase_vs_phase','level_2':'model'})
        temp_df.to_csv('final_results.csv', index=False)

        return None

# %%
data = importdata('subjects_151.data')
df = convert_to_dict(data)
# %%
model_list = (
                ('Birch', Birch(n_clusters=2, threshold=0.5)),
                ('KMeans',KMeans(n_clusters=2)),
                ('GaussianMixture',GaussianMixture(n_components=2)),
                #('Agglomerative', AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage='ward')),
)
# %%
ph5_df = {}
missing_info = [122,214,218,238,218,240,241,242,243,247,248,249,250,252,254,255,257,259,266,267,268,286,291,329,330,331,331,333,334]
for key, values in df.items():
    if 'phase5' in values.Phase.unique() and key not in missing_info:
        ph5_df[key] = values
# %%
result = execute_all(model_list, ph5_df)
# %%
results_to_dataframe(result)
# %%
len(ph5_df)
# %%
