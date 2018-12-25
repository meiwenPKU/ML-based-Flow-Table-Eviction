'''
This script is used to train and save a specified classifier
'''
import pandas as pd
import numpy as np
import time, json
import os, sys, getopt, csv
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
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

#clf = AdaBoostClassifier(n_estimators=170, learning_rate=1.1)

#clf = MLPClassifier(alpha=1e-2,hidden_layer_sizes={30}, learning_rate='adaptive', learning_rate_init=1e-2)

clf = RandomForestClassifier(max_depth=20, n_estimators=20, criterion='gini')
#clf = joblib.load("/home/yang/sdn-flowTable-management/unibs/unibs20091001-rf-uni-random-1k-10pkt.sav")
name = 'rf'

def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:N:c:p:",["ifile=", 'Npkt=', 'classifier', 'parameters'])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <tainRange> -N <Npkt> -c <classifier> -p <parameters>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -r <timeRange> -N <Npkt> -c <classifier: only be dt, rf, gnb, gbt, knn, svm, nn, ada> -p <parameters>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-N", "--Npkt"):
            Npkt = int(arg)

    x_initialTrain_file = input_file.replace("-raw-dataset.csv", "-x-inital-train-"+str(Npkt)+"pkt.npz")
    y_initialTrain_file = input_file.replace("-raw-dataset.csv", "-y-inital-train-"+str(Npkt)+"pkt.npz")
    x_val_cross_file = input_file.replace("-raw-dataset.csv", "-x-val-cross-"+str(Npkt)+"pkt.npz")
    y_val_cross_file = input_file.replace("-raw-dataset.csv", "-y-val-cross-"+str(Npkt)+'.npz')
    x_val_non_cross_file = input_file.replace("-raw-dataset.csv", "-x-val-non-cross-"+str(Npkt)+"pkt.npz")
    y_val_non_cross_file = input_file.replace("-raw-dataset.csv", "-y-val-non-cross-"+str(Npkt)+"pkt.npz")
    output_model_file = input_file.replace("-raw-dataset.csv", "-"+name+"-20t20d-"+str(Npkt)+"pkt.sav")
    scaler_file = input_file.replace("-raw-dataset.csv", "-scaler-"+str(Npkt)+"pkt.sav")

    # convert list to numpy array
    x_initialTrain = np.load(x_initialTrain_file)['arr_0']
    y_initialTrain = np.load(y_initialTrain_file)['arr_0']
    x_val_cross_npz = np.load(x_val_cross_file)
    y_val_cross_npz = np.load(y_val_cross_file)
    x_val_non_cross_npz = np.load(x_val_non_cross_file)
    y_val_non_cross_npz = np.load(y_val_non_cross_file)

    x_val_cross = [x_val_cross_npz['arr_0'], x_val_cross_npz['arr_1'], x_val_cross_npz['arr_2'], x_val_cross_npz['arr_3'], x_val_cross_npz['arr_4']]

    x_val_non_cross = [x_val_non_cross_npz['arr_0'], x_val_non_cross_npz['arr_1'], x_val_non_cross_npz['arr_2'], x_val_non_cross_npz['arr_3'], x_val_non_cross_npz['arr_4']]

    y_val_cross = [y_val_cross_npz['arr_0'], y_val_cross_npz['arr_1'], y_val_cross_npz['arr_2'], y_val_cross_npz['arr_3'], y_val_cross_npz['arr_4']]

    y_val_non_cross = [y_val_non_cross_npz['arr_0'], y_val_non_cross_npz['arr_1'], y_val_non_cross_npz['arr_2'], y_val_non_cross_npz['arr_3'], y_val_non_cross_npz['arr_4']]
    for i in range(5):
        # update the training set, merge the x_val_cross[i] and x_val_non_cross[i] with x_initialTrain

        #clf.fit(x_initialTrain, y_initialTrain)
        #y_train_pred = clf.predict(x_initialTrain)
        #conf_train = confusion_matrix(y_initialTrain, y_train_pred)
        #print conf_train
        # evaluate the cross flows
        #y_cross_predict = clf.predict(x_val_cross[i])
        #conf_val_cross = confusion_matrix(y_val_cross[i], y_cross_predict)
        #print conf_val_cross
        # evaluate the non-cross flows
        #y_non_cross_predict = clf.predict(x_val_non_cross[i])
        #conf_val_non_cross = confusion_matrix(y_val_non_cross[i], y_non_cross_predict)
        #print conf_val_non_cross
        if y_initialTrain.shape[0] == 0:
            # update the training set, merge the x_val_cross[i] and x_val_non_cross[i] with x_initialTrain
            x_initialTrain = np.copy(x_val_cross[i])
            y_initialTrain = np.copy(y_val_cross[i])
        else:
            x_initialTrain = np.concatenate((x_initialTrain, x_val_cross[i]), axis=0)
            y_initialTrain = np.concatenate((y_initialTrain, y_val_cross[i]), axis=0)
        x_initialTrain = np.concatenate((x_initialTrain, x_val_non_cross[i]), axis=0)
        y_initialTrain = np.concatenate((y_initialTrain, y_val_non_cross[i]), axis=0)
    # normalize the data
    # scaler = StandardScaler()
    # x_initialTrain = scaler.fit_transform(x_initialTrain)
    # # protocol data should not be normalized
    # mean = scaler.mean_[3]
    # std = scaler.scale_[3]
    # x_initialTrain[:,3] = (x_initialTrain[:,3]*std+mean).astype(int)

    # normalize the validation set
    clf.fit(x_initialTrain,y_initialTrain)
    joblib.dump(clf,output_model_file)
    #joblib.dump(scaler, scaler_file)

if __name__ == "__main__":
    main(sys.argv[1:])
