'''
This script is used to interpret the generated random forest model by using decision_path method and partial dependence plot
'''
from sklearn.tree import export_graphviz
from treeinterpreter import treeinterpreter as ti
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

modelFile = "/home/yang/sdn-flowTable-management/unibs/unibs20090930-rf-uni-random-1k-10pkt.sav"
forest = joblib.load(modelFile)
features = [estimator.tree_.feature for estimator in forest.estimators_]
thresholds = [estimator.tree_.threshold for estimator in forest.estimators_]
values = [estimator.tree_.value for estimator in forest.estimators_]

testLable = [1, 0, 1, 0, 0, 0, 0, 0]
testSamples = np.array([
[0.1843139999999721,0.01761625000000322,0.014050211410789975,0,-1,-1,-1,-1,-1,74,66,580,66,66],
[516.890242,0.048449000000005071,0.0,1,-1,-1,-1,-1,-1,-1,-1,-1,88,53],
[21.357391000000007,0.017856666666678695,0.025253140178792582,1,-1,-1,-1,-1,-1,-1,73,73,53,53],
[954.8317739999997,13.516587888888909,36.579160601260867,0,130,66,140,340,97,123,66,140,66,105],
[0.692819999997, 696.8735061999993, 695.5351255087157, 1, -1, -1, -1, -1, 73, 53, 71, 72, 72, 62],
[58074.419173, 12.627360333333652, 27.578380189599002, 0, 54, 58, 58, 139, 139, 139, 139, 139, 139, 139],
[0.181442000001, 41.35228166667124, 58.45410818819829, 0, -1, -1, -1, -1, -1, -1, 74, 66, 528, 66],
[56473.799893, 223.36922422222213, 316.3938339849845, 1, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62]])

# # decision paths
# indicators, n_nodes_ptr = forest.decision_path(testSamples)
#
# def findNodeId(node_id):
#     for i in range(1, len(n_nodes_ptr)):
#         if node_id < n_nodes_ptr[i]:
#             return i-1, node_id - n_nodes_ptr[i-1]
#
# print "Decision paths for test samples"
# indicators = indicators.tocoo()
# for sample_id, node_id, value in zip(indicators.row, indicators.col, indicators.data):
#     print('Rules used to predict sample %s: ' % sample_id)
#     tree_id, node_id = findNodeId(node_id)
#     if (testSamples[sample_id, features[tree_id][node_id]] <= thresholds[tree_id][node_id]):
#         threshold_sign = "<="
#     else:
#         threshold_sign = ">"
#
#     print("decision tree id %s, decision id node %s : (testSamples[%s, %s] (= %s) %s %s %s)"
#           % (tree_id, node_id,
#              sample_id,
#              features[tree_id][node_id],
#              testSamples[sample_id, features[tree_id][node_id]],
#              threshold_sign,
#              thresholds[tree_id][node_id], values[tree_id][node_id]))
#
# # tree interpreter
# print "tree interpreter"
# prediction, bias, contributions = ti.predict(forest, testSamples)
# print prediction
# print bias
# print contributions.shape
# print contributions[0,:,1]

# partial dependence plot
input_file = "/home/yang/sdn-flowTable-management/unibs/unibs20090930-dataset-uni-random-1k-10pkt.csv"
labelEncoder = "/home/yang/sdn-flowTable-management/unibs/unibs20090930-labelModel-uni-random-1k-10pkt.sav"
le = joblib.load(labelEncoder)
protocols = le.classes_

data_raw = pd.read_csv(input_file, nrows=1000)
data_raw['Protocol'] = le.transform(data_raw['Protocol'])
x = data_raw.drop('isEnd',axis=1)


features = [0, 1, 2, 14, 13]
fig, axs = plot_partial_dependence(forest, x, features,
                                   feature_names=x.columns,
                                   n_jobs=3)
fig.suptitle('Partial dependence of inactive on features\n'
             'for unibs20090930 dataset')
plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle
plt.show()
