# This script is used to generate the ROC and PR curves based on ODIN text file outputs
# To use this script:
# 1) update the global variables
# 2) run `python generate_ROC_PR_curves.py`

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

PARENT_DIRECTORY = "./Train_CIFAR-10_Test_Imagenet/" # set the parent directory of the text file results
TITLE = "ODIN vs Baseline - Train CIFAR-10, Test Imagenet - " # title of the plots

baseline_ID_data = np.loadtxt(PARENT_DIRECTORY+'confidence_Base_In.txt', delimiter=',')
baseline_OOD_data = np.loadtxt(PARENT_DIRECTORY+'confidence_Base_Out.txt', delimiter=',')
ODIN_ID_data = np.loadtxt(PARENT_DIRECTORY+'confidence_Our_In.txt', delimiter=',')
ODIN_OOD_data = np.loadtxt(PARENT_DIRECTORY+'confidence_Our_Out.txt', delimiter=',')

def generate_ROC_curve(ID_data, OOD_data, name=""):
    ID_probabilities = ID_data[:,2:]
    OOD_probabilities = OOD_data[:,2:]
    ID_labels = np.ones_like(ID_probabilities)
    OOD_labels = np.zeros_like(OOD_probabilities)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
    (fpr,tpr,thresholds) = sklearn.metrics.roc_curve(
        np.concatenate((ID_labels, OOD_labels),axis=0),
        np.concatenate((ID_probabilities, OOD_probabilities),axis=0),
    )

    roc_auc = sklearn.metrics.auc(fpr, tpr)

    # display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    # display.plot()

    # https://stackoverflow.com/questions/42894871/how-to-plot-multiple-roc-curves-in-one-plot-with-legend-and-auc-scores-in-python
    plt.plot(fpr,tpr,label="{}, AUROC={:.3f}".format(name,roc_auc))

plt.figure(0).clf()

generate_ROC_curve(baseline_ID_data,baseline_OOD_data,"baseline")
generate_ROC_curve(ODIN_ID_data,ODIN_OOD_data,"ODIN")

plt.title(TITLE + "ROC Curves")
plt.legend(loc=0)
plt.savefig(PARENT_DIRECTORY+'roc_curve.png')
plt.show()