#!/usr/bin/python3

import pickle

import numpy as np

import matplotlib.pyplot as plt


with open("predictions_jz_50000.pkl", "rb") as f:
    data = pickle.load(f)
    
for param in data.keys():
    if "y_true" in data[param].keys() and "y_pred" in data[param].keys():
        err = data[param]["y_pred"] - data[param]["y_true"]
        data[param]["MAE"] = np.mean(np.abs(err))
        data[param]["RMSE"] = np.sqrt(np.mean(err ** 2))
        print(f"{param}: MAE = {data[param]['MAE']:.4f}, RMSE = {data[param]['RMSE']:.4f}")
        
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_axisbelow(True)
ax.grid(alpha=0.25, ls='-', lw=0.5)
ax.plot(data['performance']['loss'], c="#4C72B0", lw=1.5, label='Training')
ax.plot(data['performance']['val_loss'], c="#DD8452", lw=1.5, label='Validation')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13)
ax.tick_params(axis='both', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.close()




fig, ax = plt.subplots(figsize=(8, 6))
ax.set_axisbelow(True)
ax.grid(alpha=.25, ls='-', lw=.5)
ax.scatter(data["n_atoms"]["y_true"], data["n_atoms"]["y_pred"], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label='Individual images')
ax.plot([data["n_atoms"]["y_true"].min(), data["n_atoms"]["y_true"].max()], [data["n_atoms"]["y_true"].min(), data["n_atoms"]["y_true"].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
ax.set_xlim(np.min([data["n_atoms"]["y_true"].min(), 0]), np.max([data["n_atoms"]["y_true"].max(), 5000]))
ax.set_ylim(np.min([data["n_atoms"]["y_pred"].min(), 0]), np.max([data["n_atoms"]["y_pred"].max(), 5000]))
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='both', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
ax.set_xlabel('True number of atoms', fontsize=13)
ax.set_ylabel('Predicted number of atoms', fontsize=13)
plt.tight_layout()
plt.savefig("predictions_n_atoms.png", dpi=300, bbox_inches='tight')
plt.close()



fig, ax = plt.subplots(figsize=(8, 6))
ax.set_axisbelow(True)
ax.grid(alpha=.25, ls='-', lw=.5)
ax.scatter(data["composition"]["y_true"], data["composition"]["y_pred"], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label='Individual images')
ax.plot([data["composition"]["y_true"].min(), data["composition"]["y_true"].max()], [data["composition"]["y_true"].min(), data["composition"]["y_true"].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
ax.set_xlim(np.min([data["composition"]["y_true"].min(), 0]), np.max([data["composition"]["y_true"].max(), 1]))
ax.set_ylim(np.min([data["composition"]["y_pred"].min(), 0]), np.max([data["composition"]["y_pred"].max(), 1]))
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='both', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
ax.set_xlabel('True composition', fontsize=13)
ax.set_ylabel('Predicted composition', fontsize=13)
plt.tight_layout()
plt.savefig("predictions_composition.png", dpi=300, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(8, 6))
ax.set_axisbelow(True)
ax.grid(alpha=.25, ls='-', lw=.5)
ax.scatter(data["coreshell_index"]["y_true"], data["coreshell_index"]["y_pred"], c="#4C72B0", s=4, alpha=.25, edgecolor='none', label='Individual images')
ax.plot([data["coreshell_index"]["y_true"].min(), data["coreshell_index"]["y_true"].max()], [data["coreshell_index"]["y_true"].min(), data["coreshell_index"]["y_true"].max()], c="#D62728", ls='--', lw=1.5, label='Perfect prediction')
ax.set_xlim(np.min([data["coreshell_index"]["y_true"].min(), -0.5]), np.max([data["coreshell_index"]["y_true"].max(), 2]))
ax.set_ylim(np.min([data["coreshell_index"]["y_pred"].min(), -0.5]), np.max([data["coreshell_index"]["y_pred"].max(), 2]))
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='both', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
leg = ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
leg.legend_handles[0]._sizes = [30] # agrandit le point de la légende
ax.set_xlabel('True core-shell index', fontsize=13)
ax.set_ylabel('Predicted core-shell index', fontsize=13)
plt.tight_layout()
plt.savefig("predictions_coreshell_index.png", dpi=300, bbox_inches='tight')
plt.close()