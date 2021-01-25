# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
# %%
df = pd.read_csv('all_results.csv')
# %%
def boxplot_by_data(df, metric):

    model_types = df.Model.unique()
    data_to_plot = [
        df.loc[df['Model']==i,metric] for i in model_types
        ]

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    # Custom x-axis labels
    ax.set_xticklabels(model_types)
    ax.set_ylabel(metric)
    ax.yaxis.grid(True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change fill color
        box.set(facecolor='tab:blue')

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='red', linewidth=2)

    return None

# %%
boxplot_by_data(df, 'AdjustedMutualInformation')
# %%
boxplot_by_data(df, 'AUC')
# %%
boxplot_by_data(df, 'F1')
# %%
def boxplot_all(df):
    model_types = df.Model.unique()
    data_to_plot = {}
    for element in df['Phase_vs_phase'].unique():
        data_to_plot[element] = [
            df.loc[(df['Model']==i) & (df['Phase_vs_phase']==element),'AdjustedMutualInformation'] for i in model_types
            ]
    fig = plt.figure(figsize=(10,9))

    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('Phase 1 vs Phase 2')
    bp1 = ax1.boxplot(data_to_plot.get('phase1_vs_phase2'), patch_artist=True)
    ax1.set_xticklabels([],[])
    ax1.yaxis.grid(True)

    ax2 = fig.add_subplot(2,2,2)
    bp2 = ax2.boxplot(data_to_plot.get('phase1_vs_phase3'), patch_artist=True)
    ax2.set_title('Phase 1 vs Phase 3')
    ax2.set_xticklabels([],[])

    ax3 = fig.add_subplot(2,2,3)
    bp3 = ax3.boxplot(data_to_plot.get('phase1_vs_phase4'), patch_artist=True)
    ax3.set_title('Phase 1 vs Phase 4')
    ax3.set_xticklabels(model_types, rotation=90)

    ax4 = fig.add_subplot(2,2,4)
    bp4 = ax4.boxplot(data_to_plot.get('phase1_vs_phase5'), patch_artist=True)
    ax4.set_title('Phase 1 vs Phase 5')
    ax4.set_xticklabels(model_types, rotation=90)

    ax1.set_ylabel('AMI')
    ax3.set_ylabel('AMI')
    

# %%
boxplot_all(df)
# %%
