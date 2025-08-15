import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

### Source: https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7

def plot_confusion_matrix(y_true, y_pred, labels, normalize=None, figsize=(10,10)):

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    cm_sum = np.sum(conf_matrix, axis=1, keepdims=True)
    recall = conf_matrix / cm_sum.astype(float) * 100
    prec = conf_matrix / cm_sum.astype(float) * 100
    annot = np.empty_like(conf_matrix).astype(str)
    nrows, ncols = conf_matrix.shape

    cm_sum = np.sum(conf_matrix, axis=1, keepdims=True)
    recall = conf_matrix / cm_sum.astype(float) * 100
    prec = conf_matrix /cm_sum.astype(float) * 100
    annot = np.empty_like(conf_matrix).astype(str)
    nrows, ncols = conf_matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            c = conf_matrix[i, j]
            p = recall[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = '0.0% \n0'
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

    conf_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    conf_matrix.index.name = 'Actual'
    conf_matrix.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    #sns.heatmap(conf_matrix, annot=annot, fmt='', ax=ax)
    fig.savefig("heatmap.png")
    conf_matrix.to_csv('confusion_matrix.csv')
