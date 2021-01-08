# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from time import time
import pickle
from tqdm.auto import tqdm
from sys import stdout

from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

class ResilienceStressIndex():

    def __init__(self, data):
        self.df_pc_ = self.convert_to_dict(data)
        self.df_pca_ = {}
        self.df_pc_centroids_ = {}
        self.df_pca_centroids_ = {}
        self.distances_ = {}

    def convert_to_dict(self, data):
        temp_dict = {}
        for df in data:
            temp_dict[int(df['Subject'].iloc[0][4:])] = df.copy()
            temp_dict.get(int(df['Subject'].iloc[0][4:])).reset_index(drop=True, inplace=True)

        return temp_dict

    def plot_freq(self, data_num, groups=[1,2,3,4,5], export=False):
        values = self.df_pc_.get(data_num).values
        i = 1
        # plot each column
        plt.figure(figsize=(10, 8))
        for group in groups:
            plt.subplot(len(groups), 1, i)
            plt.plot(values[:, group])
            plt.title(self.df_pc_[data_num].columns[group], y=0.5, loc='right')
            i += 1
        if export == True:
            plt.savefig('freq_plot.eps', format='eps')
        plt.show()

        return None

    def resampling_df(self, dict_copy, decimals):
        temp = {}
        with tqdm(total=len(dict_copy), file=stdout) as pbar:
            for key, data in dict_copy.items():
                pbar.set_description('Subject '+str(key))
                df_ = data.copy()
                # Change type string to float
                df_['Time'] = df_['Time'].astype('float64')
                # Create a column that rounds the time
                df_['rounded_Time'] = df_['Time'].round(decimals=decimals)
                # Group all columns by the rounded time with the median (as there are outliers in data)
                df = df_.groupby('rounded_Time', as_index=False).median()
                # Drop Time column
                df = df.drop('Time', axis=1)
                # Rename the rounded time column to Time
                df.rename(columns={'rounded_Time':'Time'}, inplace=True)
                # Transform it to int type
                df['Time'] = df['Time'].astype('int')
                # Add to dictionary
                temp[key] = df
                pbar.update(1)

        return temp
        
    def add_phase(self, data):
        for values in data.values():
            values['Phase'] = 0
            phase = range(1, values.shape[0]//120+1)
            i = 0
            for phases in phase:
                if i == 0:
                    values.loc[values['Time'] >= i, 'Phase'] = 'phase'+str(phases)
                else:
                    values.loc[values['Time'] > i, 'Phase'] = 'phase'+str(phases)
                i += 120
        
        return data

    def create_pca(self, resample=False):
        # Features to transform and implement
        variance_dict = {}
        if resample == True:
            print('Resampling data...')
            st = time()
            temp_ = self.df_pc_.copy()
            temp = self.resampling_df(temp_, decimals=0)
            temp = self.add_phase(temp)
            end = time()
            print('Finished re-sampling data in: '+str(round(end-st,2))+' seconds.')
        else:
            temp = self.df_pc_.copy()
        print('Training and transforming PCA...')

        with tqdm(total=len(temp), file=stdout) as pbar:
            for key, data in temp.items():
                pbar.set_description('Subject '+str(key))
                # Separating out the features
                x_ = data.drop(['Time','Subject','Phase'],axis=1).values
                # Transform for linearity with power transformation
                pt = PowerTransformer(method='yeo-johnson', standardize=True)
                x = pt.fit_transform(x_)
                # PCA instance
                pca = PCA(n_components=9, random_state=1)
                # Fitting and transforming the data
                principalComponents = pca.fit_transform(x)
                # Creating Dataframe of transformed data
                principalDf = pd.DataFrame(data = principalComponents, columns = ['PC'+str(n_comp) for n_comp in range(1,pca.n_components_+1)])
                # Concatenate the phase data
                finalDf = pd.concat([principalDf, data['Phase']], axis=1)
                self.df_pca_[key] = finalDf

                pbar.update(1)

        return self

    def calculate_centroids(self):
        for key, values in self.df_pc_.items():
            self.df_pc_centroids_[key] = values.groupby(by='Phase', as_index=False).mean()
        
        return self

    def calculate_centroids_pca(self):
        for key, values in self.df_pca_.items():
            self.df_pca_centroids_[key] = values.groupby(by='Phase', as_index=False).mean()  

        return self

    def plot_pca(self, subject, export=False):
        # Create layer for 2D Graph
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('PC1', fontsize = 15)
        ax.set_ylabel('PC2', fontsize = 15)
        #ax.set_title('PCA of Subject', fontsize = 20)
        # Plot
        targets = self.df_pca_.get(subject).Phase.unique()
        for target in targets:
            indicesToKeep = self.df_pca_.get(subject)['Phase'] == target
            ax.scatter(self.df_pca_.get(subject).loc[indicesToKeep, 'PC1']
                    , self.df_pca_.get(subject).loc[indicesToKeep, 'PC2']
                    , s = 50)
        ax.legend(targets)
        ax.grid()
        if export == True:
            plt.savefig('pca.eps',format='eps')
        plt.show()

        return None
    
    def plot_pca_centroid(self, subject, export=False):
        # Create layer for 2D Graph
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('PC1', fontsize = 15)
        ax.set_ylabel('PC2', fontsize = 15)
        #ax.set_title('PCA of Centroids of Subject: '+str(subject), fontsize = 20)
        # Plot
        targets = self.df_pca_centroids_.get(subject)['Phase'].unique()
        for target in targets:
            indicesToKeep = self.df_pca_centroids_.get(subject)['Phase'] == target
            ax.scatter(self.df_pca_centroids_.get(subject).loc[indicesToKeep, 'PC1']
                    , self.df_pca_centroids_.get(subject).loc[indicesToKeep, 'PC2']
                    , s = 50)
        ax.legend(targets)
        ax.grid()
        if export == True:
            plt.savefig('pca_centroid.eps',format='eps')
        plt.show()

        return None        

    def rotate_translate_axis(self):
        '''
        The function translate the first phase to the origin and rotates the second phase with the positive x-axis for centroids.
        '''
        for subject in range(len(self.df_pca_)):
            # First translation
            x = self.df_pca_[subject].PC1.loc[0]
            y = self.df_pca_[subject].PC2.loc[0]
            self.df_pca_[subject]['PC1'] = self.df_pca_[subject]['PC1'] - x
            self.df_pca_[subject]['PC2'] = self.df_pca_[subject]['PC2'] - y
            # Now get the rotation angle in rad where np.arctan(y, x) gives you the angle of rotation for all points to the positive x-axis
            angle = np.arctan2(self.df_pca_[subject].PC2.loc[1], self.df_pca_[subject].PC1.loc[1])
            for i in range(1,self.df_pca_[subject].shape[0]):
                # Calculate the new points of rotation with x'=x*cos(angle)-y*sin(angle) and y'= y*cos(angle)+x*sin(angle)
                x_ = self.df_pca_[subject].PC1.loc[i]
                y_ = self.df_pca_[subject].PC2.loc[i]
                self.df_pca_[subject].PC1.loc[i] = x_*np.cos(-1*angle)-y_*np.sin(-1*angle)
                self.df_pca_[subject].PC2.loc[i] = round(y_*np.cos(-1*angle),15)+round(x_*np.sin(-1*angle),15)

        return self

    def euclidean_distance(self):
        distances_dict = {}
        for subject in self.df_pc_centroids_.keys():
            distances_dict[subject] = [np.sqrt((self.df_pc_centroids_.get(subject).Electromyography[0]-self.df_pc_centroids_.get(subject).Electromyography[element])**2+\
                                            (self.df_pc_centroids_.get(subject).BloodVolume[0]-self.df_pc_centroids_.get(subject).BloodVolume[element])**2+\
                                            (self.df_pc_centroids_.get(subject).Breathing[0]-self.df_pc_centroids_.get(subject).Breathing[element])**2+\
                                            (self.df_pc_centroids_.get(subject).SkinConductance[0]-self.df_pc_centroids_.get(subject).SkinConductance[element])**2+\
                                            (self.df_pc_centroids_.get(subject).CorporalTemperature[0]-self.df_pc_centroids_.get(subject).CorporalTemperature[element])**2+\
                                            (self.df_pc_centroids_.get(subject).Electromyography_diff[0]-self.df_pc_centroids_.get(subject).Electromyography_diff[element])**2+\
                                            (self.df_pc_centroids_.get(subject).BloodVolume_diff[0]-self.df_pc_centroids_.get(subject).BloodVolume_diff[element])**2+\
                                            (self.df_pc_centroids_.get(subject).Breathing_diff[0]-self.df_pc_centroids_.get(subject).Breathing_diff[element])**2+\
                                            (self.df_pc_centroids_.get(subject).SkinConductance_diff[0]-self.df_pc_centroids_.get(subject).SkinConductance_diff[element])**2+\
                                            (self.df_pc_centroids_.get(subject).CorporalTemperature_diff[0]-self.df_pc_centroids_.get(subject).CorporalTemperature_diff[element])**2+\
                                            (self.df_pc_centroids_.get(subject).Electromyography_diff2[0]-self.df_pc_centroids_.get(subject).Electromyography_diff2[element])**2+\
                                            (self.df_pc_centroids_.get(subject).BloodVolume_diff2[0]-self.df_pc_centroids_.get(subject).BloodVolume_diff2[element])**2+\
                                            (self.df_pc_centroids_.get(subject).Breathing_diff2[0]-self.df_pc_centroids_.get(subject).Breathing_diff2[element])**2+\
                                            (self.df_pc_centroids_.get(subject).SkinConductance_diff2[0]-self.df_pc_centroids_.get(subject).SkinConductance_diff2[element])**2+\
                                            (self.df_pc_centroids_.get(subject).CorporalTemperature_diff2[0]-self.df_pc_centroids_.get(subject).CorporalTemperature_diff2[element])**2) for element in range(1, len(self.df_pc_centroids_.get(subject)))]
        # This will get you the dataframe
        self.distances_['euclidean'] = pd.DataFrame.from_dict(data=distances_dict, orient='index', columns=['Ph1-Ph2','Ph1-Ph3','Ph1-Ph4','Ph1-Ph5','Ph1-Ph6'])

        return self
    
    def calculate_resilience_index(self, results_):
        
        for subject_key in results_:
            resilience_index = []
            for key,values in results_.get(subject_key).items():
                # Get the max algorithm of the mean result of each fold
                resilience_index.append((key,np.max([np.mean(fold_results) for fold_results in values.values()])))
            # Append the resilience index
            self.resilience_index_[subject_key] = np.mean([resilience_index[j][1] for j in range(3,9)])-np.mean([resilience_index[i][1] for i in range(0,3)])
            # Save the complete dictionary
            self.scores_[subject_key] = resilience_index
        
        return self

    def maximum_stressor_factor(self):
        '''
        Calculates the maximum stressor factor of the sample based on the population parameter
        '''
        try:
            pop_max_stressor = self.distances_['Max_Delta_Stress'].max()
            for subject in range(len(self.distances_)):
                self.distances_['maximum_stressor_factor'] = self.distances_['Max_Delta_Stress']/pop_max_stressor
        except KeyError:
            print('Key Error: the maximum stress delta value of the samples is not calculated yet, try running the method "calculate_index" first.')
        return self
    
    def mahalanobis_distance(self):
        distances_dict = {}
        for subject in self.df_pc_.keys():
            cov = np.cov(self.df_pc_.get(subject).drop(['Time','Phase','Subject'],axis=1).values.T)
            try:
                inv_covmat = scipy.linalg.inv(cov)
                distances_dict[subject] = [scipy.spatial.distance.mahalanobis(self.df_pc_centroids_.get(subject).drop(['Phase','Time'],axis=1).loc[0].values,self.df_pc_centroids_.get(subject).drop(['Phase','Time'],axis=1).loc[element].values,inv_covmat) for element in range(1, len(self.df_pc_centroids_.get(subject)))]
            except scipy.linalg.LinAlgError:
                pass
        self.distances_['mahalanobis'] = pd.DataFrame.from_dict(data=distances_dict, orient='index', columns=['Ph1-Ph2','Ph1-Ph3','Ph1-Ph4','Ph1-Ph5','Ph1-Ph6'])
        # strange behaviour in subject 291
        self.distances_['mahalanobis'] = self.distances_.get('mahalanobis').drop(291,axis=0)

        return None
# %%
print('Importing the data...')
st = time()
with open('subjects_ksize_1001.data','rb') as data:
    df = pickle.load(data)
end = time()
print('Done importing in '+str(round(end-st,2))+' seconds.')

model = ResilienceStressIndex(df)
model.calculate_centroids()
# %% [markdown]
'''
The mean average of the percentage of explained variance of
97.6697% and a std of 1.7409% with 9 componenets. Up to 10 componenets
the changes were insignificant for this study. Mean = 96.2151
and std of 2.3353
'''
# %% [markdown]
'''
First, try to get if the data is normal for all columns.
'''
# %%
from scipy.stats import anderson

def test_normality(data):
    result = anderson(data)
    print('Statistic: {:.4f}\n'.format(result.statistic))
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.4f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('Significance level:{:.1f}%. Critical value: {:.4f}, data does not look normal (reject H0)' .format(sl, cv))
    print('\n')

    return None

features_to_test = ['Electromyography', 'BloodVolume', 'Breathing', 'SkinConductance', 'CorporalTemperature']
for column in features_to_test:
    print('Feature: '+column+'\n')
    test_normality(model.df_pc_.get(100)[column].values)
# %% [markdown]
'''
As the data is not normal we can advocate to the central limit theorem and use the Spearman correlation coefficient

## VIF (Variable Inflation Factors)

https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/
'''
# %%
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
for subject in [414,440,437]:
    print('Subject: {}'.format(subject))
    print(calc_vif(model.df_pc_.get(subject).drop(['Time','Phase','Subject'],axis=1)))
# %% [markdown]
'''
As many subjects present multicolinearity in at least one variable we can justify
the implementation of PCA and euclidean distance.

As a next step we can get manhaladabian distance
'''
# %%
model.plot_freq(data_num=414, export=False)
# %%
model.create_pca(resample=False)
# %% [markdown]
'''
The PCA with four components has a explained variance bigger than 90%
with five componenets no much difference can be notice.
'''
# %%
model.calculate_centroids_pca()
# %% [markdown]
'''
441(0.7271), 440(0.6918), 437(0.6357), 425(0.7121), 424(0.6476), 415(0.6525), 414(0.7436), 318 (0.5920)
Ploting one subject pca centroids of each phase
'''
# %%
model.plot_pca(subject=414, export=False)
model.plot_pca_centroid(414, export=False)
# %% [markdown]
'''
Now we can calculate the distances
'''
# %% [markdown]
'''
# ## Calculating distances
'''
# %%
model.euclidean_distance()
# %%
model.mahalanobis_distance()
# %% [markdown]
'''
We need to check correlation between variables and exclude them in the euclidean distance
in the self.df_pc_centroids.get(subject)

The variables that are correlated are with >0.25
Breathing - Electromyography 0.66
Corporal Temperature - Electromyography 0.67
BloodVolumePulse - Corporal Temperature 0.6
SkinConductanve - Blood Volume 0.4
Skin Conductance - Electromyography 0.32

1. Check normality
2. Check Pearson or Spearman
3. Check VIF for correlation
4. Now you can create PCA with the justification of euclidean distance
4b. Or you can use manhalanobis distance with set (phase1) vs centroids of each phase

To avoid loose of data:
1. Create cluster phases and compare Phase 1 vs phase 2, phase 1 vs phase 3, etc.
2. Get an index that gets the level of homogenity or heterogeneity of the clusters.

'''
# %%
import seaborn as sns

def calculate_index(distances):
    distances['Max_Delta_Stress'] = np.nan
    distances['Recovery_Delta_Last_Phase'] = np.nan
    distances['Resilience_Index'] = np.nan
    # Calculate of maximum point of stress before the last stressor phase and get the recovery phase delta
    for element in distances.index.values:
        if pd.isnull(distances['Ph1-Ph6'].loc[element]) == True:
            if pd.isnull(distances['Ph1-Ph5'].loc[element]) == True:
                if pd.isnull(distances['Ph1-Ph4'].loc[element]) == True:
                    distances['Max_Delta_Stress'].loc[element] = distances['Ph1-Ph2'].loc[element]
                    distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph3'].loc[element]
            else:
                distances['Max_Delta_Stress'].loc[element] = distances[['Ph1-Ph2','Ph1-Ph4']].loc[element].max()
                distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph5'].loc[element]
        else:
            distances['Max_Delta_Stress'].loc[element] = distances[['Ph1-Ph2','Ph1-Ph4']].loc[element].max()
            distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph6'].loc[element]
    # Get the resilience index, the more positive the better resilience to stress.
    distances['Resilience_Index'][(distances['Max_Delta_Stress'] != distances['Recovery_Delta_Last_Phase'])] = distances['Max_Delta_Stress']-distances['Recovery_Delta_Last_Phase']
    distances['Resilience_Index'][(distances['Max_Delta_Stress'] == distances['Recovery_Delta_Last_Phase'])] = -1*distances['Recovery_Delta_Last_Phase']

    sns.distplot(distances['Resilience_Index'], hist=True, kde=True, color='darkblue',hist_kws={'edgecolor':'black'})
    plt.title('Density Plot and Histogram of Resilience Index')
    plt.ylabel('Density')
    plt.xlabel('Resilience Index')
    plt.show()

    sns.boxplot(x='Resilience_Index', orient='h',data=distances)
    plt.title('Boxplot of Resilience Index')
    plt.show()

    return distances
# %%
regular_index = calculate_index(model.distances_.copy())
regular_index
# %%
def calculate_index_recovery_factor(distances):
    '''
    Calculates an index value between 0 and 1, the closer to 0 the worst, the closer to 1 the more resilient.
    '''
    # Initialize column variables
    distances['Max_Delta_Stress'] = np.nan
    distances['Recovery_Delta_Last_Phase'] = np.nan
    distances['Resilience_Index'] = np.nan   
    # Calculate the max delta stress of each individual and its recovery delta of the last recovery phase
    for element in distances.index.values:
        if pd.isnull(distances['Ph1-Ph6'].loc[element]) == True:
            if pd.isnull(distances['Ph1-Ph5'].loc[element]) == True:
                if pd.isnull(distances['Ph1-Ph4'].loc[element]) == True:
                    distances['Max_Delta_Stress'].loc[element] = distances['Ph1-Ph2'].loc[element]
                    distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph3'].loc[element]
            else:
                distances['Max_Delta_Stress'].loc[element] = distances[['Ph1-Ph2','Ph1-Ph4']].loc[element].max()
                distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph5'].loc[element]
        else:
            distances['Max_Delta_Stress'].loc[element] = distances[['Ph1-Ph2','Ph1-Ph4']].loc[element].max()
            distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph6'].loc[element]
    # Get the maximum value of the population of delta stress
    population_max_delta_stress = distances['Max_Delta_Stress'].max()
    # Get the resilience index, the more positive the better resilience to stress.
    distances['Resilience_Index'] = distances['Max_Delta_Stress']**2/(population_max_delta_stress*distances['Recovery_Delta_Last_Phase'])

    print('Population_Max_Delta_Stress: %.4f'%(population_max_delta_stress))
    sns.distplot(distances['Resilience_Index'], hist=True, kde=True, color='darkblue',hist_kws={'edgecolor':'black'})
    plt.title('Density Plot and Histogram of Resilience Index')
    plt.ylabel('Density')
    plt.xlabel('Resilience Index')
    plt.show()

    sns.boxplot(x='Resilience_Index', orient='h',data=distances)
    plt.title('Boxplot of Resilience Index')
    plt.show()

    return distances
# %%
recovery_factor = calculate_index_recovery_factor(model.distances_)
recovery_factor
# %%
recovery_factor.sort_values('Resilience_Index',axis=0,ascending=True)
# %%
def calculate_index_modified(distances):
    distances['Max_Delta_Stress'] = np.nan
    distances['Recovery_Delta_Last_Phase'] = np.nan
    distances['Resilience_Index'] = np.nan
    # Calculate of maximum point of stress before the last stressor phase and get the recovery phase delta
    for element in distances.index.values:
        if pd.isnull(distances['Ph1-Ph6'].loc[element]) == True:
            if pd.isnull(distances['Ph1-Ph5'].loc[element]) == True:
                if pd.isnull(distances['Ph1-Ph4'].loc[element]) == True:
                    distances['Max_Delta_Stress'].loc[element] = distances['Ph1-Ph2'].loc[element]
                    distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph3'].loc[element]
            else:
                distances['Max_Delta_Stress'].loc[element] = distances[['Ph1-Ph2','Ph1-Ph4']].loc[element].max()
                distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph5'].loc[element]
        else:
            distances['Max_Delta_Stress'].loc[element] = distances[['Ph1-Ph2','Ph1-Ph4']].loc[element].max()
            distances['Recovery_Delta_Last_Phase'].loc[element] = distances['Ph1-Ph6'].loc[element]
    # Get the maximum value of the population of delta stress
    population_max_delta_stress = distances['Max_Delta_Stress'].max()
    # Get the resilience index, the more positive the better resilience to stress.
    distances['Resilience_Index'] = (distances['Max_Delta_Stress']-distances['Recovery_Delta_Last_Phase'])/population_max_delta_stress

    sns.distplot(distances['Resilience_Index'], hist=True, kde=True, color='darkblue',hist_kws={'edgecolor':'black'})
    plt.title('Density Plot and Histogram of Resilience Index')
    plt.ylabel('Density')
    plt.xlabel('Resilience Index')
    plt.show()

    sns.boxplot(x='Resilience_Index', orient='h',data=distances)
    plt.title('Boxplot of Resilience Index')
    plt.show()

    return distances
# %%
dist_modi = calculate_index_modified(model.distances_)
dist_modi.sort_values('Resilience_Index',axis=0,ascending=True)
# %%
dist_mahal = calculate_index_modified(model.distances_.get('mahalanobis'))
dist_mahal.sort_values('Resilience_Index',axis=0,ascending=True)
# %%
model.plot_freq(376, [1,2,3,4,5])
# %% [markdown]
# You need to consider that powertransformation does a normalization and that FIRST you need to transform then scale.

# %%

