# %% 
#Importing Libraries
import pandas as pd
import numpy as np
import time
# visualizations
import seaborn as sns
rs = 42
np.random.seed(seed = rs)
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#Display all rows in the notebook
pd.options.display.max_rows = 20
# Sklearn Models and Evaluation
from sklearn import metrics, utils
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score as R2
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.model_selection import cross_val_score,  cross_validate
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
#Multiclass
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
#Statsmodel
from scipy import stats
# %%
def nulls(df):
    tot = df.isnull().sum().sort_values(ascending = False)
    per = df.isnull().sum()/df.isnull().count().sort_values(ascending = False)
    missing_cont = pd.concat([tot, per] , axis = 1 , keys = ['Total', 'Percentage'])
    print(missing_cont)
# %%
def uniques(df):
    print("Unique Value Count:")
    cols = df.columns.tolist()
    for col in cols:
        print(col + " = " + str(len(df[col].unique())))
# %%
def cv_report_ovr(X,y, classifier, cnames, model_name, scale = False):
    kf =KFold(n_splits=10, random_state=rs, shuffle = True)
    # define X and y
    y = y.values
    X = X.values
    X, y = utils.shuffle(X, y, random_state=rs)
    auc_scores = []
    acc_scores = []
    f1_scores = []
    kappa_scores = []
    tn= []
    tp = []
    fn = []
    fp = []
    # Binarize the output with 4 classes
    y = label_binarize(y, classes=[0,1,2,3])
    st = time.time()
    for train_index, test_index in kf.split(X):
        # Split train-test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if scale == True:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        model = OneVsRestClassifier(classifier)
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        auc = [roc_auc_score(y_test[:,i], y_proba[:,i]) for i in range(4)]
        kappa = [metrics.cohen_kappa_score(y_test[:,i], y_pred[:,i]) for i in range(4)]
        f1 = [metrics.f1_score(y_test[:,i], y_pred[:,i]) for i in range(4)]
        acc = [metrics.accuracy_score(y_test[:,i], y_pred[:,i]) for i in range(4)]
        # Append the scores to the list
        auc_scores.append(auc)
        f1_scores.append(f1)
        kappa_scores.append(kappa)
        acc_scores.append(acc)
        cm = metrics.multilabel_confusion_matrix(y_test, y_pred)
        tp.append(cm[0][0])
        fp.append(cm[1][0])
        tn.append(cm[1][1])
        fn.append(cm[0][1])
    end = time.time()
    TP = int(np.sum(tp))
    FN = int(np.sum(fn))
    TN = int(np.sum(tn))
    FP = int(np.sum(fp))
    matrix = np.array([[TN, FP] , [FN, TP]])
    measured_time = end-st
    print(f'Model: {model_name}')
    print(f' AUC : {np.average(auc_scores)}')
    print(f' F1 weighted : {np.average(f1_scores)}')
    print(f' Accuracy : {np.average(acc_scores)}')
    #print(f' Kappa Statitic : {np.average(kappa_scores)}')
    kappa_scores = 0
    print('Process Complete in : '+str(measured_time)+' sec.')
    #matrix_norm = np.array([[round(TN/sum_up,3), round(FP/sum_up,3)] , [round(FN/sum_down,3), round(TP/sum_down,3)]])
    return (matrix,cnames), auc_scores, f1_scores, acc_scores, kappa_scores,model_name ,measured_time        

    
# %% 
def cv_report_4classes(X,y, classifier, cnames, model_name, scale = False):
    kf =KFold(n_splits=10, random_state=rs, shuffle = True)
    # define X and y
    y = y.values
    X = X.values
    X, y = utils.shuffle(X, y, random_state=rs)
    auc_scores = []
    acc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    kappa_scores = []
    fir = []
    sec = []
    thi = []
    fou = []
    fiv = []
    six = []
    sev = []
    eig = []
    nin = []
    ten = []
    ele = []
    twe = []
    tht = []
    fot = []
    fit = []
    sit = []
    # Binarize the output with 4 classes
    st = time.time()
    for train_index, test_index in kf.split(X):
        # Split train-test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if scale == True:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        model_ovr = OneVsRestClassifier(classifier)
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class = 'ovr')
        f1 = metrics.f1_score(y_test, y_pred, average = 'macro')
        precision = metrics.precision_score(y_test, y_pred, average = 'macro')
        recall = metrics.recall_score(y_test, y_pred, average = 'macro')
        kappa = metrics.cohen_kappa_score(y_test,y_pred)
        acc = metrics.accuracy_score(y_test,y_pred)
        # Append the scores to the list
        auc_scores.append(auc)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        kappa_scores.append(kappa)
        acc_scores.append(acc)
        cm = metrics.confusion_matrix(y_test, y_pred)
        fir.append(cm[0][0])
        sec.append(cm[1][0])
        thi.append(cm[2][0])
        fou.append(cm[3][0])
        fiv.append(cm[0][1])
        six.append(cm[1][1])
        sev.append(cm[2][1])
        eig.append(cm[3][1])
        nin.append(cm[0][2])
        ten.append(cm[1][2])
        ele.append(cm[2][2])
        twe.append(cm[3][2])
        tht.append(cm[0][3])
        fot.append(cm[1][3])
        fit.append(cm[2][3])
        sit.append(cm[3][3])
    end = time.time()
    fir = int(np.sum(fir))
    sec = int(np.sum(sec))
    thi = int(np.sum(thi))
    fou = int(np.sum(fou))
    fiv = int(np.sum(fiv))
    six = int(np.sum(six))
    sev = int(np.sum(sev))
    eig = int(np.sum(eig))
    nin = int(np.sum(nin))
    ten = int(np.sum(ten))
    ele = int(np.sum(ele))
    twe = int(np.sum(twe))
    tht = int(np.sum(tht))
    fot = int(np.sum(fot))
    fit = int(np.sum(fit))
    sit = int(np.sum(sit))
    matrix = np.array([[fir, fiv, nin, tht] , [sec, six, ten, fot], [thi, sev, ele, fit], [fou, eig, twe, sit]])
    measured_time = end-st
    print(f'Model: {model_name}')
    print(f' AUC : {np.average(auc_scores)}')
    print(f' F1 macro: {np.average(f1_scores)}')
    print(f' Precision macro: {np.average(precision_scores)}')
    print(f' Recall macro: {np.average(recall_scores)}')
    print(f' Accuracy : {np.average(acc_scores)}')
    print(f' Kappa Statitic : {np.average(kappa_scores)}')
    print('Process Complete in : '+str(measured_time)+' sec.')
    #matrix_norm = np.array([[round(TN/sum_up,3), round(FP/sum_up,3)] , [round(FN/sum_down,3), round(TP/sum_down,3)]])
    return (matrix,cnames), auc_scores, f1_scores, precision_scores, recall_scores, acc_scores, kappa_scores,model_name ,measured_time

# %%
def cross_validation_report(X,y, model, cnames, model_name, scale = False):
    kf =KFold(n_splits=10, random_state=rs, shuffle = True)
    # define X and y
    y = y.values
    X = X.values
    X, y = utils.shuffle(X, y, random_state=rs)

    auc_scores = []
    acc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    kappa_scores = []
    tn= []
    tp = []
    fn = []
    fp = []
    st = time.time()
    for train_index, test_index in kf.split(X):
        # Split train-test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if scale == True:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        model = model.fit(X_train, y_train)
        cm = confusion_matrix(y_test, model.predict(X_test))
        # Append to auc_scores the auc of the model
        y_proba = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test) 
        auc_scores.append(roc_auc_score(y_test,y_proba))
        f1_scores.append(metrics.f1_score(y_test,y_pred, average = 'weighted'))
        precision_scores.append(metrics.precision_score(y_test,y_pred, average = 'weighted'))
        recall_scores.append(metrics.recall_score(y_test,y_pred, average = 'weighted'))
        kappa_scores.append(metrics.cohen_kappa_score(y_test,y_pred))
        acc_scores.append(metrics.accuracy_score(y_test,y_pred))
        tn.append(cm[0][0])
        fn.append(cm[1][0])
        tp.append(cm[1][1])
        fp.append(cm[0][1])
    end = time.time()
    TP = int(np.sum(tp))
    FN = int(np.sum(fn))
    TN = int(np.sum(tn))
    FP = int(np.sum(fp))
    measured_time = end-st
    print(f'Model: {model_name}')
    print(f' AUC : {np.average(auc_scores)}')
    print(f' F1 weighted : {np.average(f1_scores)}')
    print(f' Precision : {np.average(precision_scores)}')
    print(f' Recall : {np.average(recall_scores)}')
    print(f' Accuracy : {np.average(acc_scores)}')
    print(f' Kappa Statitic : {np.average(kappa_scores)}')
    print('Process Complete in : '+str(measured_time)+' sec.')
    matrix = np.array([[TN, FP] , [FN, TP]])
    #matrix_norm = np.array([[round(TN/sum_up,3), round(FP/sum_up,3)] , [round(FN/sum_down,3), round(TP/sum_down,3)]])
    return (matrix,cnames), auc_scores, f1_scores, precision_scores, recall_scores, acc_scores, kappa_scores,model_name ,measured_time
# %% 
#Print confusion matrix
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,10),fontsize = 14, model_name = 'Model'):
    df_cm = pd.DataFrame(confusion_matrix, index = class_names, columns = class_names)
    try: 
      plt.figure()
      heatmap = sns.heatmap(df_cm, annot= True, cmap = "Blues",fmt = '0.0f')
    except ValueError: 
        raise ValueError('Confusion Matrix values must be integers')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} Confusion Matrix')
    return plt
# %% 
# %% 
from statsmodels.stats.outliers_influence import variance_inflation_factor
#Variable Inflation Factors
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
def plot_loss_comparison(preds):
    overall_loss_comparison = preds[~preds.quantile_loss.isnull()].\
      pivot_table(index='method', values='quantile_loss').\
      sort_values('quantile_loss')
    # Show overall table.
    print(overall_loss_comparison)
  
    # Plot overall.
    with sns.color_palette("tab10", 1):
        ax = overall_loss_comparison.plot.barh()
        plt.title('Total Quant Loss', loc='left')
        sns.despine(left=True, bottom=True)
        plt.xlabel('Quantile loss')
        plt.ylabel('')
        ax.legend_.remove()
  
    # Per quantile.
    per_quantile_loss_comparison = preds[~preds.quantile_loss.isnull()].\
        pivot_table(index='q', columns='method', values='quantile_loss')
    # Sort by overall quantile loss.
    per_quantile_loss_comparison = \
        per_quantile_loss_comparison[overall_loss_comparison.index]
    print(per_quantile_loss_comparison)
  
    # Plot per quantile.
    with sns.color_palette('tab10'):
        ax = per_quantile_loss_comparison.plot.barh()
        plt.title('Quantile loss per quantile', loc='left')
        sns.despine(left=True, bottom=True)
        handles, labels = ax.get_legend_handles_labels()
        plt.xlabel('Quantile loss')
        plt.ylabel('Quantile')
        # Reverse legend.
        ax.legend(reversed(handles), reversed(labels))
# %% 
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as smapi
class QRWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
    def fit(self, X_tr, y_tr, q):
        self.q=q
        if self.fit_intercept:
            X_tr = smapi.add_constant(X_tr)
        self.model_ = QuantReg(y_tr,X_tr)
        self.result_ = self.model_.fit(q=self.q , max_iter=1000)
        return self
    def predict(self, X_te):
        if self.fit_intercept:
            X_te = smapi.add_constant(X_te)
        return self.result_.predict(X_te)
    def get_params(self, deep = False):
        return {'fit_intercept':self.fit_intercept}
    def params(self):
        return self.result_.params
    def summary(self):
        print(self.result_.summary())
        
# %% 
def prsquared(y, y_pred, q): #Koenker R2 for quantiles
    q = q
    endog = y
    e = endog - y_pred
    #try: 
        #e = endog - y_pred
    #except: 
        #print(endog.shape)
        #print(y_pred.shape)
    e = np.where(e < 0, (1 - q) * e, q * e)
    e = np.abs(e)
    ered = endog - stats.scoreatpercentile(endog, q * 100)
    ered = np.where(ered < 0, (1 - q) * ered, q * ered)
    ered = np.abs(ered)
    return 1 - np.sum(e) / np.sum(ered)

def cv_score_quantreg(X,y , model, cv , q = 0.1):
    # Initialize local variables
    results = []
    kf5 = KFold(n_splits=cv, shuffle=False)
    # Define X and y
    X = X.values
    y = y.values
    # Start the manual cross validation process
    for train_index, test_index in kf5.split(X):
        # Split the train and test of each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit the pipeline
        model.fit(X_train, y_train)
        # Get the prediction
        y_pred = model.predict(X_test)
        # Get the score and append it
        score = prsquared(y_test , y_pred, q = q)
        results.append(score)
    return results

def quantile_loss(q, y, f):
    # q: Quantile to be evaluated, e.g., 0.5 for median.
    # y: True value.
    # f: Fitted or predicted value.
    e = y - f
    return np.maximum(q * e, (q - 1) * e)
def mqloss(y_true, y_pred, q):  
  if (q > 0) and (q < 1):
    residual = y_true - y_pred 
    return np.mean(np.maximum(q * residual, residual * (q - 1)))
  else:
    return np.nan

def compute_quantile_loss(y_true, y_pred, quantile):
    """
    
    Parameters
    ----------
    y_true : 1d ndarray
        Target value.
        
    y_pred : 1d ndarray
        Predicted value.
        
    quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.
    """
    residual = y_true - y_pred
    return np.average(np.maximum(quantile * residual, (quantile - 1) * residual))


def cross_validation_report_quant_reg(X,y, model, model_name, quantile = 1, scale = False, log_transform = False, qr = True, cv = 10):
    kf =KFold(n_splits=cv, random_state=rs, shuffle = True)
    # define X and y
    y = y.values
    X = X.values
    X, y = utils.shuffle(X, y, random_state=rs)
    q_loss_scores = []
    pseudo_r2_scores = []
    
    if log_transform == True:
        model = TransformedTargetRegressor(regressor=model,                  
        func=np.log, inverse_func=np.exp)
    st = time.time()
    for train_index, test_index in kf.split(X):
    # Split train-test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
      #Scale data
        if scale == True:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        model = model.fit(X_train, y_train, q = quantile)
        

        #predict
        y_pred = model.predict(X_test)
        # Append scores
        pseudo_r2_scores.append(prsquared(y_test , y_pred, q = quantile))
        q_loss_scores.append(compute_quantile_loss(y_test, y_pred, quantile = quantile))
    end = time.time()
    measured_time = end-st
    print('Process Complete in : '+str(measured_time)+' sec.')
    print(f'Model: {model_name}')
    print(f' Pseudo R2 : {np.average(pseudo_r2_scores)}')
    print(f' Quantile Loss : {np.average(q_loss_scores)}')
    return q_loss_scores, pseudo_r2_scores,measured_time,model_name
# %%
def test_holdout(X_ho, y_ho, X_tr, y_tr, model , quantile = 0.5, qr = True):
    if qr == True:
        model.fit(X_tr, y_tr , q= quantile)
    else:
        model.fit(X_tr, y_tr)
    y_pred = model.predict(X_ho)
    pseudo_r2_score = prsquared(y_ho, y_pred, q = quantile)
    loss = mqloss(y_ho,y_pred, q = qr)
    print(f'pseudo_r2 : {pseudo_r2_score}')
    print(f'Quantile Loss : {loss}')
    return pseudo_r2_score, loss, y_pred

def plot_real_predict(y_test,y_pred, q , model_name):

    fig, axs = plt.subplots(3, figsize=(10,10))
    fig.suptitle( str(model_name) +' Predicted vs. Real ' + str(q) + ' Quantile', size=14, y=1.02)
    DFyy = pd.DataFrame({'y_test':y_test,'y_pred': y_pred})
    DFyy.sort_values(by=['y_test'],inplace=True)
    axs[0].plot(np.arange(0,len(DFyy),1), DFyy['y_pred'], color  = 'blue', label = 'Predicted')
    axs[0].plot(np.arange(0,len(DFyy),1), DFyy['y_test'], alpha=0.5, color = 'green', label  = 'Real')
    axs[0].set_ylabel('Current Income')
    axs[0].set_xlabel('Index ')
    axs[0].legend()
    print('Observations were sorted by y_test values, i.e. higher index => higher Income value')
    axs[1].scatter(y_test, y_pred, c = 'blue')
    axs[1].plot(y_test, y_test, "green")
    axs[1].set_xlabel('y_actual')
    axs[1].set_ylabel('y_predicted')
    axs[2].plot(np.arange(0,len(y_test),1), y_test, 'g.', markersize=10, label='Actual')
    axs[2].plot(np.arange(0,len(y_test),1), y_pred, 'b-', label='Prediction', alpha =0.5)
    axs[2].set_xlabel('Observations')
    axs[2].set_ylabel('Current Income')
    axs[2].legend(loc='upper right')
    return plt.show()

# %%
def cross_validation_report_reg(X,y, model, model_name,  scale = False, log_transform = False):
    kf =KFold(n_splits=10, random_state=rs, shuffle = True)
    # define X and y
    y = y.values
    X = X.values
    X, y = utils.shuffle(X, y, random_state=rs)
    mae_scores = []
    r2_scores = []
    adjusted_r2 = []
    rmse_scores = []
    if log_transform == True:
        model = TransformedTargetRegressor(regressor=model,                  
        func=np.log, inverse_func=np.exp)
    st = time.time()
    for train_index, test_index in kf.split(X):
    # Split train-test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
      #Scale
        if scale == True:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        model = model.fit(X_train, y_train)
        # Append scores
        y_pred = model.predict(X_test)

        r2_scores.append(metrics.r2_score(y_test,y_pred))
        adjusted_r2.append(1 - (1-metrics.r2_score(y_test,y_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
        )
        rmse_scores.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
        mae_scores.append(metrics.mean_absolute_error(y_test,y_pred))
    
    end = time.time()
    measured_time = end-st
    print('Process Complete in : '+str(measured_time)+' sec.')
    print(f'Model: {model_name}')
    print(f' R2 : {np.average(r2_scores)}')
    print(f'Adjusted R2: {np.average(adjusted_r2)}')
    print(f' MAE : {np.average(mae_scores)}')
    print(f'RMSE: {np.average(rmse_scores)}')
    return mae_scores, r2_scores, adjusted_r2, rmse_scores ,measured_time,model_name

# %%
def boxplot_metrics(metric_values, subtitle ,xlabel  = 'Classifiers',ylabel = 'AUC',
                    limits = (0.4,1)):
    labels = metric_values.keys()
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    plt.title(subtitle)
    plt.xlabel(xlabel)
    plt.xticks(rotation = 45)
    plt.ylabel(ylabel)
    plt.ylim(limits)
    box = plt.boxplot(metric_values.values(), patch_artist = True)
    colors = ['blue', 'orchid','peru', 'green', 'teal','wheat','red', 'indigo', 'brown', 'pink',
            'grey','magenta','limegreen','lightsteelblue','royalblue','navy','slateblue','aquamarine','tan','lightcoral',
            'salmon','khaki','darkslategray','lawngreen','purple','azure','darkseagreen','rosybrown','orange']
    for b in box:
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
            patch.set_alpha(0.1)

    ax.set_xticklabels(labels)
    return plt



# %%
def time_bar_chart(metric_values, subtitle = 'Classifiers Train Time',xlabel = 'Classifiers',ylabel= 'seg'):
    labels = metric_values.keys()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111)
    fig.suptitle(subtitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    bar = plt.bar(labels, metric_values.values())
    ax.set_xticklabels(labels)
    return plt
# %%
def test_validation_bin(clf, X_train, y_train, X_holdout, selected =False, scale = False):
    if scale == True: 
        pipe = Pipeline([('scaler', StandardScaler()), ('model', clf)])
        model =pipe.fit(X_train, y_train)
        predictions = model.predict(X_holdout)
        proba = model.predict_proba(X_holdout)[:,1]
    if selected == False:
        model = clf.fit(X_train, y_train)
        predictions = model.predict(X_holdout)
        proba = model.predict_proba(X_holdout)[:,1]
    else:
        model = clf.fit(X_train[selected], y_train)
        predictions = model.predict(X_holdout[selected])
        proba = model.predict_proba(X_holdout[selected])[:,1]
    return predictions, proba

def test_validation_mult(clf, X_train, y_train, X_holdout, selected =False , ovr = False):
    if ovr == True: 
        model = OneVsRestClassifier(clf)
    if selected == False:
        model = model.fit(X_train, y_train)
        predictions = model.predict(X_holdout)
        proba = model.predict_proba(X_holdout)
    else:
        model = model.fit(X_train[selected], y_train)
        predictions = model.predict(X_holdout[selected])
        proba = model.predict_proba(X_holdout[selected])
    return predictions, proba

def test_validation_reg(reg, X_train, y_train, X_holdout, selected =False, scale = False):
    if scale == True: 
        pipe = Pipeline([('scaler', StandardScaler()), ('model', reg)])
        model =pipe.fit(X_train, y_train)
        predictions = model.predict(X_holdout)
    if selected == False:
        model = reg.fit(X_train, y_train)
        predictions = model.predict(X_holdout)
    else:
        model = reg.fit(X_train[selected], y_train)
        predictions = model.predict(X_holdout[selected])
    return predictions
# %%
def multi_algo_auc_plot(proba_list, y_test1, y_test2,  title ,linear = ['LogReg']):
    # dictionaries for roc curve output
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc = {}
    n_alg = len(proba_list.keys())

    for num,i in enumerate(proba_list.keys()): 
        if i in linear:
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test2, proba_list[i], pos_label=1)
        else:
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test1, proba_list[i], pos_label=1)

    # auc scores
    for num,i in enumerate(proba_list.keys()): 
        if i in linear:
            roc_auc[i] = roc_auc_score(y_test2, proba_list[i])
        else:
            roc_auc[i] = roc_auc_score(y_test1, proba_list[i])

    # matplotlib
    plt.figure(figsize = (8,6))
    plt.style.use('seaborn')
    lw = 2
    colors = ['orange', 'teal','purple', 'green', 'teal','wheat','red', 'indigo', 'brown', 'pink',
                'grey','magenta','limegreen','lightsteelblue','royalblue','navy','slateblue','aquamarine','tan','lightcoral',
                'salmon','khaki','darkslategray','lawngreen','purple','azure','darkseagreen','rosybrown','orange']
    for i, color in zip(proba_list.keys(), colors):
        plt.plot(fpr[i], tpr[i],  color = color, lw = lw, label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    # title
    plt.title(title, fontsize = 14) 
    # x label
    plt.xlabel('False Positive Rate', fontsize = 14)
    # y label
    plt.ylabel('True Positive rate', fontsize = 14) 

    plt.legend(loc='best' , fontsize = 14) 
    plt.savefig('ROC',dpi=300)
    return plt.show()
# %%
def multi_class_auc_plot(clf, X_train, y_train, X_test, y_test, title = 'Multiclass LGBM OVR ROC curve', labels = ['Q1','Q2','Q3','Q4']):
    clf = OneVsRestClassifier(clf) 
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pred_prob = clf.predict_proba(X_test)
    # dictionaries for roc curve output
    fpr = {}
    tpr = {}
    thresh ={}
    roc_auc = {}
    n_class =len(y_train.value_counts())

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #Plot
    plt.figure(figsize = (8,6))
    lw = 2
    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'magenta','green'])
    for i, color in zip(range(n_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(labels[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.title(title, fontsize = 14) 
    plt.xlabel('False Positive Rate', fontsize = 14) 
    plt.ylabel('True Positive rate', fontsize = 14) 
    plt.legend(loc='best', fontsize = 14) 
    plt.savefig('Multiclass ROC',dpi=300)
    return plt.show()


# %%
# Testing function for R reticulate
def sum5(x):
    return x + 5