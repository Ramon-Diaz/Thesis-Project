import pandas as pd
import numpy as np
import os 
from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from time import time
import pickle
import multiprocessing as mp
        
def importdata(folderName):
    print('Importing Data...')
    st = time()
    # Set working directory
    os.chdir(folderName)
    # Import all cases from the text files
    filenames = sorted(glob('caso*.txt'))
    df = [pd.read_csv(f, sep=',', skiprows=6, encoding = 'ISO-8859-1', low_memory=False) for f in filenames]
    # Add an identifier to each dataframe
    for element in range(len(df)):
        df[element]['Subject'] = 'case'+str(filenames[element][4:].split('.')[0])
        df[element].drop(df[element].index[0], inplace=True)
        df[element].reset_index(drop=True,inplace=True)
        df[element]['Time'] = df[element]['Time'].astype(float)
        df[element] = df[element].rename(columns={'C: SC':'SkinConductance','D: Conductancia de la piel':'SkinConductance','A: BVP':'BloodVolume','B: Volumen del pulso sanguineo':'BloodVolume','B: Temp':'CorporalTemperature','E: Temperatura':'CorporalTemperature','D: RA':'Breathing','C: Respiracion':'Breathing','E: RT':'Electromyography','A: Electromiografia':'Electromyography'})
    os.chdir('..') # Return to the working directory
    end = time()
    print('Finished importing in '+str(round(end-st,2))+' seconds.')

    return df

def add_phase_column(df_pc):
    print('Adding phase column...')
    st = time()
    # Add a phase feature that indicates the phase it belongs
    for subject in range(len(df_pc)):
        df_pc[subject]['Phase'] = ''
        phases = range(1, int(df_pc[subject]['Time'].iloc[-1])//120+1)
        i = 0.0
        for phase in phases:
            df_pc[subject].loc[(df_pc[subject]['Time']>=i) & (df_pc[subject]['Time']<(i+120)),'Phase'] = 'phase'+str(phase)
            i+=120
        df_pc[subject].loc[df_pc[subject].index[-1], 'Phase'] = 'phase'+str(phases[-1])
    end = time()
    print('Finished adding phase column in '+str(round(end-st,2))+' seconds.')

    return df_pc

def fit_scaler(df_pc_filtered, scaler=StandardScaler()):
    print('Fitting the scaler...')
    st = time()
    # Concatenate complete dataframe
    df_all = pd.concat(df_pc_filtered)
    # Separating out the features
    features = ['Electromyography', 'BloodVolume', 'Breathing', 'SkinConductance', 'CorporalTemperature']
    new_features = [element+'_diff' for element in features]
    X = df_all.loc[:, features+new_features].values
    scaler.fit(X)
    end = time()
    print('Finished fitting the scaler in '+str(round(end-st,2))+' seconds.')

    return scaler
    
def transform_scaler(df_pc_filtered, scaler):
    print('Scaling data...')
    st = time()
    features = ['Electromyography', 'BloodVolume', 'Breathing', 'SkinConductance', 'CorporalTemperature']
    new_features = [element+'_diff' for element in features]
    for subject in range(len(df_pc_filtered)):
        try:
            df_pc_filtered[subject][features+new_features] = scaler.transform(df_pc_filtered[subject][features+new_features])
        except AttributeError:
            raise AttributeError('Try fitting the scaler first.')
    end = time()
    print('Finished scaling the data in '+str(round(end-st,2))+' seconds.')

    return df_pc_filtered

def add_nontime_dependencies(df_pc_filtered):
    print('Adding non-time dependencies...')
    st = time()

    features = ['Electromyography', 'BloodVolume', 'Breathing', 'SkinConductance', 'CorporalTemperature']
    new_features = [element+'_diff' for element in features]
    for i in features:
        new_features.append(i+'_diff2')
    for subject in range(len(df_pc_filtered)):
        for name in new_features:
            df_pc_filtered[subject][name] = 0.0

        for column in new_features:
            if column[-1] == '2':
                df_pc_filtered[subject][column] = df_pc_filtered[subject][column.split('_')[0]] - df_pc_filtered[subject][column.split('_')[0]].shift(2)
                df_pc_filtered[subject][column].fillna(method='bfill',inplace=True)
            else:    
                df_pc_filtered[subject][column] = df_pc_filtered[subject][column.split('_')[0]] - df_pc_filtered[subject][column.split('_')[0]].shift(1)
                df_pc_filtered[subject][column].fillna(method='bfill',inplace=True)
            
    end = time()
    print('Finished in '+str(round(end-st,2))+' seconds.')
    
    return df_pc_filtered        

def export_pickle(df_pc_filtered):
    print('Exporting data...')
    with open('non_scaled_subjects.data','wb') as data:
        pickle.dump(df_pc_filtered, data)
    print('DONE!!')

    return None

def filter_column(x, k_size, type):
    '''
    Input: x = array to transform
            k = size of the kernel
            type = type of filter median or mean
    Output: array of filtered column. The boundaries are calculated by repeating the endpoint.
    '''
    assert k_size % 2 == 1, 'Kernel size must be odd.'

    k2 = (k_size - 1) // 2
    y = np.zeros ((len (x), k_size), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    if type == 'median':
        return np.median(y, axis=1)
    else:
        return np.mean(y, axis=1)
        
def apply_filter(df_pc, subject, k_size, type='median'):
    print('Subject: '+str(subject))
    df_filtered = df_pc[subject].copy()
    # Eliminate the first 0.1 sec
    df_filtered = df_filtered.loc[500:]
    assert type == 'mean' or type=='median', 'ERROR: Not a valid command, try mean or median.'
    # Apply filter by column
    features = ['Electromyography', 'BloodVolume', 'Breathing', 'SkinConductance', 'CorporalTemperature']
    for column in features:
        df_filtered[column] = filter_column(df_filtered[column].values, k_size, type)

    return df_filtered
    
if __name__ == "__main__":
    st_all = time()

    df = importdata('ProComp')
    df = add_phase_column(df)
    pool = mp.Pool(10)
    print('Applying Median Filter...')
    st = time()
    df_filtered = pool.starmap_async(apply_filter, [(df, i, 151) for i in range(len(df))]).get()
    pool.close()
    end = time()
    print('Time: '+str(round((end-st)/60,2))+' minutes.')
    df_filtered = add_nontime_dependencies(df_filtered)
    scaler = fit_scaler(df_filtered, StandardScaler())
    df_filtered = transform_scaler(df_filtered, scaler)

    end_all = time()
    print('Done.\nTime: '+str(round((end_all-st_all)/60,2))+' minutes.')
    export_pickle(df_filtered)