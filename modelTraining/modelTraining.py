from __future__ import division
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.externals import joblib
from sklearn import preprocessing
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
input_file = parent_path+"/univ1/dataset-uni-ts-1k.csv"
output_model = parent_path + "/univ1/gbc-uni-1k.sav"
rng = np.random.RandomState(0)
isTest = True
# #############################################################################
# read the data
data_raw = pd.read_csv(input_file)
print data_raw.shape
# encoding categorical features
le = preprocessing.LabelEncoder()
data_raw['Protocol'] = le.fit_transform(data_raw['Protocol'])
# ############################################################################
clf = RandomForestClassifier(n_estimators=30,max_depth=22)
#clf = GradientBoostingClassifier(learning_rate=0.01,n_estimators=60, max_depth=10)
if isTest:
    # split the data into train set and test set
    train_set, test_set = train_test_split(data_raw, test_size = .2)
    #X_train = train_set.drop(['pktActiveTime','avgReason','stdReason'],axis = 1)
    X_train = train_set.drop(['isEnd'],axis=1)
    y_train = train_set['isEnd']
    print np.bincount(y_train)
    #X_test = test_set.drop(['pktActiveTime','avgReason','stdReason'],axis = 1)
    X_test = test_set.drop(['isEnd'],axis=1)
    y_test = test_set['isEnd']
    print np.bincount(y_test)
    clf.fit(X_train,y_train)
    y_train_pred = clf.predict(X_train)
    conf_train = confusion_matrix(y_train,y_train_pred)
    print ('train: accuracy of active flow entry is %s, accuracy of inactive flow entry is %s' % (conf_train[0,0]/float(conf_train[0,0]+conf_train[0,1]),conf_train[1,1]/float(conf_train[1,0]+conf_train[1,1])))
    print ('train: precision=%s' % (precision_score(y_train, y_train_pred)))
    y_predict = clf.predict(X_test)
    conf_test = confusion_matrix(y_test,y_predict)
    print ('test: accuracy of active flow entry is %s, accuracy of inactive flow entry is %s' % (conf_test[0,0]/float(conf_test[0,0]+conf_test[0,1]),conf_test[1,1]/float(conf_test[1,0]+conf_test[1,1])))
    print ('test: precision=%s' % (precision_score(y_test,y_predict)))

else:
    X = data_raw.drop('isEnd',axis=1)
    y = data_raw['isEnd']
    clf.fit(X,y)
    joblib.dump(clf,output_model)
