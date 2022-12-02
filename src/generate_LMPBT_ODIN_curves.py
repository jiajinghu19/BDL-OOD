# This script is used to generate the ROC and PR curves based on ODIN text file outputs
# To use this script:
# 1) update the global variables
# 2) run `python generate_ROC_PR_curves.py`

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

ID = "FashionMNIST"
OOD = "MNIST"

IMAGE_FILE_PREFIX = "./LMPBT_vs_ODIN_{}-ID_{}-OOD_".format(ID,OOD)
LMPBT_DIRECTORY = "./LMPBT/results/Train_{}/".format(ID) # set the parent directory of the results
ODIN_DIRECTORY = "./ODIN/Train_{}_Test_{}/".format(ID,OOD)
TITLE = "LMPBT vs ODIN \n Trained {}, Test {} (ID) and {} (OOD) \n ".format(ID,ID,OOD) # title of the plots


LMPBT_ID_nll = np.load(LMPBT_DIRECTORY + 'in_distro_nll_vae.npy') # LMPBT's ID negative log likelihoods
LMPBT_OOD_nll = np.load(LMPBT_DIRECTORY + 'out_distro_nll_vae.npy') # LMPBT's OOD negative log likelihoods
ODIN_ID_probabilities = np.loadtxt(ODIN_DIRECTORY+'confidence_Our_In.txt', delimiter=',') # ODIN's ID probabilities
ODIN_OOD_probabilities = np.loadtxt(ODIN_DIRECTORY+'confidence_Our_Out.txt', delimiter=',') # ODIN's OOD probabilities

def process_LMPBT_data(ID_data_unprocessed, OOD_data_unprocessed):
    return (
        np.concatenate((np.ones(len(ID_data_unprocessed)), np.zeros(len(OOD_data_unprocessed)))), # array of actual labels
        np.concatenate((ID_data_unprocessed, OOD_data_unprocessed)) # array of preditected values
    )

def process_ODIN_data(ID_data_unprocessed, OOD_data_unprocessed):
    ID_probabilities = ID_data_unprocessed[:,2:]
    OOD_probabilities = OOD_data_unprocessed[:,2:]
    ID_labels = np.ones_like(ID_probabilities)
    OOD_labels = np.zeros_like(OOD_probabilities)

    return (
        np.concatenate((ID_labels, OOD_labels),axis=0), # array of actual labels
        np.concatenate((ID_probabilities, OOD_probabilities),axis=0) # array of preditected values
    )

def calc_ROC_PR_data(labels, predictions, pos_label=1):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
    (fpr,tpr,thresholds) = roc_curve( labels, predictions, pos_label=pos_label )

    (precision,recall,thresholds) = precision_recall_curve( labels, predictions, pos_label=pos_label )

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)


    fpr95 = (fpr[np.argmin(np.abs(tpr - 0.95))].min())

    return fpr, tpr, roc_auc, precision, recall, pr_auc, fpr95

(LMPBT_labels, LMPBT_predictions) = process_LMPBT_data(LMPBT_ID_nll, LMPBT_OOD_nll)
(
  LMPBT_fpr, LMPBT_tpr, LMPBT_auroc,
  LMPBT_precision, LMPBT_recall, LMPBT_aupr, LMPBT_fpr95
) = calc_ROC_PR_data(LMPBT_labels, LMPBT_predictions, pos_label=0)

(ODIN_labels, ODIN_predictions) = process_ODIN_data(ODIN_ID_probabilities, ODIN_OOD_probabilities)
(
  ODIN_fpr, ODIN_tpr, ODIN_auroc,
  ODIN_precision, ODIN_recall, ODIN_aupr, ODIN_fpr95
) = calc_ROC_PR_data(ODIN_labels, ODIN_predictions)

plt.figure(1).clf()
plt.title(TITLE + "ROC Curves")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(LMPBT_fpr,LMPBT_tpr,label="LMPBT, AUROC={:.3f}".format(LMPBT_auroc))
plt.plot(ODIN_fpr,ODIN_tpr,label="ODIN, AUROC={:.3f}".format(ODIN_auroc))
plt.legend(loc=0)
plt.savefig(IMAGE_FILE_PREFIX+'roc_curve.png')

plt.figure(2).clf()
plt.title(TITLE + "PR Curves")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(LMPBT_recall,LMPBT_precision,label="LMPBT, AUPR={:.3f}".format(LMPBT_aupr))
plt.plot(ODIN_recall,ODIN_precision,label="ODIN, AUPR={:.3f}".format(ODIN_aupr))
plt.legend(loc='best')
plt.savefig(IMAGE_FILE_PREFIX+'pr_curve.png')

plt.show()