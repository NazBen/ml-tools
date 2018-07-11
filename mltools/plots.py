import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_importance(importance, features, ax=None):
    features = np.asarray(features)
    idx = importance.argsort()
    if ax is None:
        figsize = (4, int(len(features)/3))
    else:
        figsize = None
    pd.DataFrame(importance[idx], index=features[idx]).plot(
        kind='barh', figsize=figsize, ax=ax)


def plot_correlation(df, method='pearson', ratio=(0.7, 0.5)):
    n_features = df.shape[1]
    fig, ax = plt.subplots(figsize=(ratio[0]*n_features, ratio[1]*n_features))
    corr = df.corr(method=method)
    sns.heatmap(corr, ax=ax, vmin=-1., vmax=1., square=True,
                robust=True, annot=True, fmt='.2f')
    ax.set_title('Correlation map')
    fig.tight_layout()