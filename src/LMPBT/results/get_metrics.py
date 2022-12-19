## borrowed from src/LMPBT/LMPBT_fork/get_metrics.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PARENT_DIRECTORY = "./Train_FashionMNIST/" # set the parent directory of the results
TITLE = "LMPBT - Train Fashion MNIST, Test MNIST - " # title of the plots

nll_in_distro = np.load(PARENT_DIRECTORY + 'in_distro_nll_vae.npy')
nll_out_distro = np.load(PARENT_DIRECTORY + 'out_distro_nll_vae.npy')

k = 10
idx = np.argpartition(nll_out_distro, k)[:k]
print('nll_out_distro idx', nll_out_distro[idx])
idx = np.argpartition(nll_in_distro, k)[:k]
print('nll_in_distro idx', nll_in_distro[idx])

combined = np.concatenate((nll_in_distro, nll_out_distro))
labels = np.concatenate((np.ones(len(nll_in_distro)), np.zeros(len(nll_out_distro))))

(fpr, tpr, thresholds) = roc_curve(labels, combined, pos_label=0)
(precision,recall,thresholds) = precision_recall_curve(labels, -combined)

roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)
fpr80 = (fpr[np.argmin(np.abs(tpr - 0.8))].min())

print ("AUROC:", roc_auc, "AUPR:",pr_auc, "FPR80:",fpr80)


plt.figure(1).clf()
plt.title(TITLE + "ROC Curves")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr,tpr,label="LMPBT, AUROC={:.3f}".format(roc_auc))
plt.legend(loc=0)
plt.savefig(PARENT_DIRECTORY+'roc_curve.png')

plt.figure(2).clf()
plt.title(TITLE + "PR Curves")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall,precision,label="LMPBT, AUPR={:.3f}".format(pr_auc))
plt.legend(loc='best')
plt.savefig(PARENT_DIRECTORY+'pr_curve.png')




plt.show()