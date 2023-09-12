# %%
import os
import sys
import random
import open3d as o3d
import numpy as np
import pandas as pd
import torch
from plyfile import PlyData, PlyElement
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import copy
import matplotlib.pyplot as plt
from sklearn import manifold
import pickle
# %%
# f = open('scene0500_01_feature_maps.pkl','rb')
folder_path = "/nfs/nas-6.1/ynjuan/v1/visualize"
all_features = []
all_labels = []
count = 0
for idx, name in enumerate(os.listdir(folder_path)):
    if 'feature' in name:
        count += 1
        with open(os.path.join(folder_path, name), 'rb') as f:
            feature_map = pickle.load(f)
            features = feature_map['feature_map'].cpu().numpy().tolist()
            labels = feature_map['target'].cpu().numpy().tolist()
            all_features += features
            all_labels += labels
# %%
# features = pickle.load(f)
# print(features)
# print(features.shape)
# %%
X_tsne = manifold.TSNE(n_components=2, init='random', verbose=1).fit_transform(features)
# %%
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
# %%
fix, ax = plt.subplots()
for label in np.unique(labels):
    i = np.where(labels == label)
    ax.scatter(X_norm[i, 0 ], X_norm[i, 1], label=label, s=20)
ax.legend(loc=3, prop={'size': 6})
plt.axis('off')
plt.savefig('tsne.jpg')
plt.show()
# %%
