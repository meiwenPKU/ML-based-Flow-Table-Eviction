'''
This script is used to find the importance of different features
'''
from sklearn.tree import export_graphviz
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

modelFile = "/home/yang/sdn-flowTable-management/univ1/rf-uni-random-1k-10pkt.sav"
forest = joblib.load(modelFile)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
features = ['idle','mean','std','pro','p1','p2','p3','p4','p5', 'p6', 'p7', 'p8', 'p9', 'p10']
indices = np.argsort(importances)[::-1]
features = [features[i] for i in indices]

# Print the feature ranking
print("Feature ranking:")

for f in range(forest.n_features_):
    print("%d. feature %s (%f)" % (f + 1, features[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(forest.n_features_), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(forest.n_features_), features)
plt.xlim([-1, forest.n_features_])
plt.show()
