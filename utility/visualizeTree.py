'''
This script is used to visualize one decision tree from a random forest model
'''
from sklearn.tree import export_graphviz
from sklearn.externals import joblib

modelFile = "/home/yang/sdn-flowTable-management/unibs/unibs20090930-rf-uni-random-1k-5pkt.sav"
rf = joblib.load(modelFile)
estimator = rf.estimators_[0]

# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
                feature_names = ['idle','mean','std','pro','p1','p2','p3','p4','p5'],
                class_names = ['active','inactive'],
                rounded = True, proportion = False,
                precision = 2, filled = True,
                max_depth = 8)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
