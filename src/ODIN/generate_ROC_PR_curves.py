# This script is used to generate the ROC and PR curves based on ODIN text file outputs
# To use this script:
# 1) update the global variables
# 2) run `python generate_ROC_PR_curves.py`

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PARENT_DIRECTORY = "./Train_SVHN_Test_CIFAR-10/" # set the parent directory of the text file results
TITLE = "ODIN vs Baseline - Train SVHN, Test CIFAR-10 - " # title of the plots

baseline_ID_data = np.loadtxt(PARENT_DIRECTORY+'confidence_Base_In.txt', delimiter=',')
baseline_OOD_data = np.loadtxt(PARENT_DIRECTORY+'confidence_Base_Out.txt', delimiter=',')
ODIN_ID_data = np.loadtxt(PARENT_DIRECTORY+'confidence_Our_In.txt', delimiter=',')
ODIN_OOD_data = np.loadtxt(PARENT_DIRECTORY+'confidence_Our_Out.txt', delimiter=',')

def calc_ROC_PR_data(ID_data, OOD_data):
    ID_probabilities = ID_data[:,2:]
    OOD_probabilities = OOD_data[:,2:]
    ID_labels = np.ones_like(ID_probabilities)
    OOD_labels = np.zeros_like(OOD_probabilities)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
    (fpr,tpr,thresholds) = roc_curve(
        np.concatenate((ID_labels, OOD_labels),axis=0),
        np.concatenate((ID_probabilities, OOD_probabilities),axis=0),
    )

    (precision,recall,thresholds) = precision_recall_curve(
        np.concatenate((ID_labels, OOD_labels),axis=0),
        np.concatenate((ID_probabilities, OOD_probabilities),axis=0),
    )

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)


    # display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    # display.plot()

    return fpr, tpr, roc_auc, precision, recall, pr_auc
    
(
  baseline_fpr, baseline_tpr, baseline_auroc,
  baseline_precision, baseline_recall, baseline_aupr
) = calc_ROC_PR_data(baseline_ID_data, baseline_OOD_data)
(
  ODIN_fpr, ODIN_tpr, ODIN_auroc,
  ODIN_precision, ODIN_recall, ODIN_aupr
) = calc_ROC_PR_data(ODIN_ID_data, ODIN_OOD_data)

plt.figure(1).clf()
plt.title(TITLE + "ROC Curves")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(baseline_fpr,baseline_tpr,label="Baseline, AUROC={:.3f}".format(baseline_auroc))
plt.plot(ODIN_fpr,ODIN_tpr,label="ODIN, AUROC={:.3f}".format(ODIN_auroc))
plt.legend(loc=0)
plt.savefig(PARENT_DIRECTORY+'roc_curve.png')

plt.figure(2).clf()
plt.title(TITLE + "PR Curves")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(baseline_recall,baseline_precision,label="Baseline, AUPR={:.3f}".format(baseline_aupr))
plt.plot(ODIN_recall,ODIN_precision,label="ODIN, AUPR={:.3f}".format(ODIN_aupr))
plt.legend(loc='best')
plt.savefig(PARENT_DIRECTORY+'pr_curve.png')




plt.show()