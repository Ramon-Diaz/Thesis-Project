# %%
import pickle
from time import time
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from sys import stdout

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

def get_results(df, model):
    temp = df.copy()
    test_df = (
                    ('phase1_vs_phase2',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase2')]),
                    ('phase1_vs_phase3',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase3')]),
                    ('phase1_vs_phase4',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase4')]),
                    ('phase1_vs_phase5',temp[(temp['Phase'] == 'phase1') | (temp['Phase'] == 'phase5')]),
                )
    for dataset in test_df:
        pass

    return None

def train_cluster(dataset, model):
    print('Training the model...')
    st = time()
    # Initialize the variables
    scores = {}
    # Split x and y as we know the labels
    X = dataset.drop(['Time','Subject','Phase'],axis=1).values
    y = dataset['Phase'].values
    # Fit the model and return the labels
    y_pred = model.fit_predict(X)
    # Compute the AUC score
    scores['auc'] = roc_auc_score(y, y_pred)
    scores['f1'] = f1_score(y, y_pred)
    end = time()
    print('Done importing in {:.2f} seconds.'.format(end-st))

    return scores

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import roc_auc_score, f1_score

# %%
data = importdata('subjects_151.data')
df = convert_to_dict(data)

model_list = (
                ('AffinityPropagation', AffinityPropagation(verbose=True)),
                ('Agglomerative', AgglomerativeClustering(n_clusters=2)),
)
# %%
result = train_cluster(df.get(100)[(df.get(100)['Phase'] == 'phase1') | (df.get(100)['Phase'] == 'phase2')], model_list[1][1])
# %%
