import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


IN_DISTRO_RESULTS = 'svhn100_nll_vae.npy'
OUT_DISTRO_RESULTS = 'cifar100_nll_vae.npy'

nll_in_distro = np.load(IN_DISTRO_RESULTS)
nll_out_distro = np.load(OUT_DISTRO_RESULTS)
combined = np.concatenate((nll_in_distro, nll_out_distro))
labels = np.concatenate((np.ones(len(nll_in_distro)), np.zeros(len(nll_out_distro))))
fpr, tpr, thresholds = metrics.roc_curve(labels, combined, pos_label=0)
# plot_roc_curve(fpr, tpr)
rocauc = metrics.auc(fpr, tpr)
aucprc = metrics.average_precision_score(labels, -combined)
fpr80 = (fpr[np.argmin(np.abs(tpr - 0.8))].min())

print ("AUROC:", rocauc, "AUPR:",aucprc, "FPR80:",fpr80)
