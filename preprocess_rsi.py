# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 22})

import os 
from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import pickle
import time
import multiprocessing as mp

from tqdm.auto import tqdm
from sys import stdout

from scipy import fft

class PreprocessRSI():

    def __init__(self, folderName):
        self.folderName_ = folderName
        self.features_ = ['Electromyography', 'BloodVolume', 'Breathing', 'SkinConductance', 'CorporalTemperature']
        self.df_pc_filtered_ = []

        self.df_pc_ = self.importdata()
        self.add_phase_column()
        

    def importdata(self):
        print('Importing Data...')
        st = time.time()
        # Set working directory
        os.chdir(self.folderName_)
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
        end = time.time()
        print('Finished importing in '+str(round(end-st,2))+' seconds.')

        return df

    def add_phase_column(self):
        print('Adding phase column...')
        st = time.time()
        # Add a phase feature that indicates the phase it belongs
        for subject in range(len(self.df_pc_)):
            self.df_pc_[subject]['Phase'] = ''
            phases = range(1, int(self.df_pc_[subject]['Time'].iloc[-1])//120+1)
            i = 0.0
            for phase in phases:
                self.df_pc_[subject].loc[(self.df_pc_[subject]['Time']>=i) & (self.df_pc_[subject]['Time']<(i+120)),'Phase'] = 'phase'+str(phase)
                i+=120
            self.df_pc_[subject].loc[self.df_pc_[subject].index[-1], 'Phase'] = 'phase'+str(phases[-1])
        end = time.time()
        print('Finished adding phase column in '+str(round(end-st,2))+' seconds.')

        return self

    def plot_freq(self, data, data_num, groups=[1,2,3,4,5], export=False):
        values = data[data_num].values
        i = 1
        # plot each column
        plt.figure(figsize=(10, 8))
        for group in groups:
            plt.subplot(len(groups), 1, i)
            plt.plot(values[:, group])
            plt.title(data[data_num].columns.values[group], y=0.5, loc='right')
            plt.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
            i += 1
        if export == True:
            plt.savefig('freq_plot.eps', format='eps')
        plt.show()

        return None

    def filter_column(self, x, k_size, type):
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
            
    def apply_filter(self, subject, k_size, type='median'):
        print('Subject: '+str(subject))
        df_filtered = self.df_pc_[subject].copy()
        # Eliminate the first 0.1 sec
        df_filtered = df_filtered.loc[500:]
        assert type == 'mean' or type=='median', 'ERROR: Not a valid command, try mean or median.'
        # Apply filter by column
        for column in self.features_:
            df_filtered[column] = self.filter_column(df_filtered[column].values, k_size, type)

        return df_filtered

    def fit_scaler(self, scaler=StandardScaler()):
        print('Fitting the scaler...')
        st = time.time()
        # Concatenate complete dataframe
        df_all = pd.concat(self.df_pc_filtered_)
        # Separating out the features
        X = df_all.loc[:, self.features_+self.new_features_].values
        self.scaler_ = scaler
        self.scaler_.fit(X)
        end = time.time()
        print('Finished fitting the scaler in '+str(round(end-st,2))+' seconds.')

        return self
    
    def transform_scaler(self):
        print('Scaling data...')
        st = time.time()
        for subject in range(len(self.df_pc_filtered_)):
            try:
                self.df_pc_filtered_[subject][self.features_+self.new_features_] = self.scaler_.transform(self.df_pc_filtered_[subject][self.features_+self.new_features_])
            except AttributeError:
                raise AttributeError('Try fitting the scaler first.')
        end = time.time()
        print('Finished scaling the data in '+str(round(end-st,2))+' seconds.')

        return self

    def add_nontime_dependencies(self):
        print('Adding non-time dependencies...')
        st = time.time()
        self.new_features_ = [element+'_diff' for element in self.features_]
        for i in self.features_:
            self.new_features_.append(i+'_diff2')
        for subject in range(len(self.df_pc_filtered_)):
            for name in self.new_features_:
                self.df_pc_filtered_[subject][name] = 0.0

            for column in self.new_features_:
                if column[-1] == '2':
                    self.df_pc_filtered_[subject][column] = self.df_pc_filtered_[subject][column.split('_')[0]] - self.df_pc_filtered_[subject][column.split('_')[0]].shift(2)
                    self.df_pc_filtered_[subject][column].fillna(method='bfill',inplace=True)
                else:    
                    self.df_pc_filtered_[subject][column] = self.df_pc_filtered_[subject][column.split('_')[0]] - self.df_pc_filtered_[subject][column.split('_')[0]].shift(1)
                    self.df_pc_filtered_[subject][column].fillna(method='bfill',inplace=True)
                
        end = time.time()
        print('Finished in '+str(round(end-st,2))+' seconds.')
        
        return self
    
    def execute_median_filter(self, k_size):
        print('Applying Median Filter...')
        st = time.time()
        # Initialize the pool
        pool = mp.Pool(mp.cpu_count())
        #with tqdm(total=len(self.df_pc_), file=stdout, position=0, leave=True) as pbar:
        #    for i in tqdm(range(len(self.df_pc_)), position=0, leave=True, desc='  Subject'):
                #pbar.set_description('  Subject')
        #        self.df_pc_filtered_.append(self.apply_filter(i,k_size))
        #        pbar.update(1)
        self.df_pc_filtered_ = pool.starmap_async(self.apply_filter, [(i, k_size) for i in range(len(self.df_pc_))]).get()
        pool.close()
        end = time.time()
        print('Time: '+str(round((end-st)/60,2))+' minutes.')

        return self

    def export_pickle(self):
        print('Exporting data...')
        with open('subjects.data','wb') as data:
            pickle.dump(self.df_pc_filtered_, data)
        print('DONE!!')

        return self

# %%
if __name__ == "__main__":
    st_all = time.time()

    model = PreprocessRSI('ProComp')
    #model.plot_freq(model.df_pc_, 57, groups=[1,2,3,4,5], export=False)
    model.execute_median_filter(151)
    model.add_nontime_dependencies()
    model.fit_scaler(StandardScaler())
    model.transform_scaler()

    end_all = time.time()
    print('Done.\nTime: '+str(round((end_all-st_all)/60,2))+' minutes.')
    model.export_pickle()
# %%
'''
from scipy.signal import welch
from scipy import fft

SAMPLE_RATE = 256 # sample rate by second
variable_to_measure = 'Breathing'
model.plot_freq(414)
data = model.df_pc_.get(414).loc[(model.df_pc_.get(414)['Phase']=='phase1'),variable_to_measure]
N = data.size # the number of observations
yf = fft.rfft(data.values)
xf = fft.rfftfreq(N, 1/SAMPLE_RATE) # to plot the frequencies of the sample rate
power = np.abs(yf)**2
#xf, power = welch(data.values, fs = 1/(SAMPLE_RATE/N))

#plt.plot(xf, power)
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Power')
# Get the positive peak
#pos_mask = np.where(xf > 0)
#freqs = xf[pos_mask]
#peak_freq = freqs[power[pos_mask].argmax()]
# Get the peak
peak_freq = xf[power.argmax()]
#plt.show()
'''
# %%
