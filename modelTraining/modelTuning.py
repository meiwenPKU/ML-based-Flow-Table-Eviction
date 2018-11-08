'''
This script use k-fold cross validation to tune the hyperparameters for different machine learning algorithms,
including svm, decision tree, random forest, ada boosting, gradient boosting tree, and shallow neural network.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFECV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.externals import joblib
from datetime import datetime
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import os
from collections import Counter
from sklearn import preprocessing

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
input_file = parent_path+"/unibs/unibs20091001-dataset-uni-random-4k-10pkt.csv"

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GradientBoostingClassifier()]
names = ["Linear SVM", "RBF/Poly SVM","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","GBR"]

tuned_parameters=[
    {'clf__kernel':['linear'],'clf__C':[1e-1,1e0,1e1]},
    {'clf__kernel':['rbf'],'clf__C':[1e0,1e1],'clf__gamma':[1e0]},
    {'clf__criterion':['entropy','gini'],'clf__max_depth':[10,15,20,25,30]},
    {'clf__criterion':['gini'],'clf__max_depth':[25],'clf__n_estimators':[20]},
    {'clf__hidden_layer_sizes':[(30),(15,15),(10,10,10)],'clf__alpha':[1e-2],'clf__learning_rate': ['invscaling'],'clf__learning_rate_init':[1e-2]},
    {'clf__n_estimators':[20,40,60,100,200],'clf__learning_rate':[0.01, 0.1, 1]},
    {'clf__loss':['deviance','exponential'],'clf__learning_rate':[0.01], 'clf__n_estimators':[60],'clf__max_depth':[15,20,25,30]}]

#load the data from csv file
data_raw = pd.read_csv(input_file)
print "raw data shape = "
print data_raw.shape
print Counter(data_raw['Protocol'])
#encoding categorical features
le = preprocessing.LabelEncoder()
data_raw['Protocol'] = le.fit_transform(data_raw['Protocol'])
#split the data into train set and test set
train_set,test_set = train_test_split(data_raw,test_size=.2, random_state=40)
X_train_set = train_set.drop('isEnd',axis=1)
y_train_set = train_set['isEnd']
print np.bincount(y_train_set)
print Counter(train_set['Protocol'])

for parameters, clf, name in zip(tuned_parameters, classifiers,names):
    # if not (name == "Random Forest"):
    #     continue
    print "cv on classifier %s" % (name)
    estimators = [('standardization', StandardScaler()),('clf',clf)]
    pipe = Pipeline(estimators)
    clf_gs = GridSearchCV(pipe, param_grid=parameters, cv=5, n_jobs=2, verbose=3, scoring='precision')
    clf_gs.fit(X_train_set, y_train_set)

    means = clf_gs.cv_results_['mean_test_score']
    stds = clf_gs.cv_results_['std_test_score']
    score_times = clf_gs.cv_results_['mean_score_time']
    for mean, std, scoretime, params in zip(means, stds, score_times, clf_gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f), score time=%s, for %r" % (mean, std * 2, scoretime, params))
