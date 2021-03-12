# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import normaltest
from scipy.stats import shapiro

from autorank import autorank, create_report, plot_stats, latex_table

# %%
def import_data():
    df_euclidean = pd.read_csv('Results/euclidean_distances_pca.csv', index_col=0).drop('Ph1-Ph6', axis=1)
    df_silhouette = pd.read_csv('Results/silhouette_distances.csv', index_col=0)
    df_mahalanobis = pd.read_csv('Results/mahalanobis_distances.csv', index_col=0)

    return df_euclidean, df_mahalanobis, df_silhouette

# %%
df_euclidean, df_mahalanobis, df_silhouette = import_data()
# %%

rsi = pd.DataFrame()
rsi['Euclidean'] = df_euclidean.Resilience_Index
rsi['Mahalanobis'] = df_mahalanobis.Resilience_Index
rsi['Silhouette'] = df_silhouette.Resilience_Index

af = pd.DataFrame()
af['Euclidean'] = df_euclidean.stress_factor
af['Mahalanobis'] = df_mahalanobis.stress_factor
af['Silhouette'] = df_silhouette.stress_factor

# %% [markdown]
'''

'''

# %%
pd.set_option('display.max_columns', 7)

res = autorank(rsi, alpha=0.05, verbose=False)
print(res)
create_report(res)
plot_stats(res, allow_insignificant=True)
plt.show()
latex_table(res)
# %%
res1 = autorank(af, alpha=0.05, verbose=False)
print(res1)
create_report(res1)
plot_stats(res1, allow_insignificant=True)
plt.show()
latex_table(res1)
# %%
