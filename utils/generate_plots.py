from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

def calc_ROC_PR_data(ID_probabilities, OOD_probabilities, args):
    ID_labels = np.ones_like(ID_probabilities)
    OOD_labels = np.zeros_like(OOD_probabilities)

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

    plt.figure(1).clf()
    plt.title( "ROC Curves")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr,tpr,label="VAE, AUROC={:.3f}".format(roc_auc))
    plt.legend(loc=0)
    plt.savefig(f'data/curves/{args.filename_prefix}-{args.train_data}_roc_curve.png')

    plt.figure(2).clf()
    plt.title("PR Curves")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall,precision,label="VAE, AUPR={:.3f}".format(pr_auc))
    plt.legend(loc='best')
    plt.savefig(f'data/curves/{args.filename_prefix}-{args.train_data}_pr_curve.png')
    return fpr, tpr, roc_auc, precision, recall, pr_auc
