'''
This script is used to apply different under sampling method to generate and save balanced dataset
'''
'''
This script is used to tune the hyperparameters for different classical machine learning algorithms. Different modelTuning.py, this script applies 5-fold cross validation on a rolling basis given that the flow data is time seriless. Here cross validation on a rolling basis means:

---------------------
| train       | val |
---------------------
---------------------------
| train             | val |
---------------------------
---------------------------------
| train                   | val |
---------------------------------

Each validation part spans 10% for the whole time duration

This script supports to pass a dictionary as command line arguments, also use under sampling to deal with imbalanced dataset
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
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)

supportSamplers = {'cc':    ClusterCentroids(random_state=0),
"enn":    EditedNearestNeighbours(),
"renn":     RepeatedEditedNearestNeighbours(),
"aknn":   AllKNN(allow_minority=True),
"cnn":    CondensedNearestNeighbour(random_state=0),
"oss":    OneSidedSelection(random_state=0),
"ncr":     NeighbourhoodCleaningRule()}

def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:s:N:",["ifile=", 'sampler', 'Npkt'])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -s <sampler> -N <Npkt>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -s <sampler> -N <Npkt>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ('-s', "--sampler"):
            sampler_name = arg
        elif opt in ('-N', "--Npkt"):
            Npkt = int(arg)
    if sampler_name not in supportSamplers:
        print "%s is not supported, we only support cluster Centroids, edit Nearest neighbors, repeated edit Nearest neighbors, all knn, condense Nearest neighbors, one side selection, neighbourhood clean rule" % sampler_name
        exit()

    x_initialTrain_file = input_file.replace("-raw-dataset.csv", "-x-inital-train-"+str(Npkt)+"pkt.npz")
    y_initialTrain_file = input_file.replace("-raw-dataset.csv", "-y-inital-train-"+str(Npkt)+"pkt.npz")
    x_val_cross_file = input_file.replace("-raw-dataset.csv", "-x-val-cross-"+str(Npkt)+"pkt.npz")
    y_val_cross_file = input_file.replace("-raw-dataset.csv", "-y-val-cross-"+str(Npkt)+'.npz')
    x_val_non_cross_file = input_file.replace("-raw-dataset.csv", "-x-val-non-cross-"+str(Npkt)+"pkt.npz")
    y_val_non_cross_file = input_file.replace("-raw-dataset.csv", "-y-val-non-cross-"+str(Npkt)+"pkt.npz")
    x_balanced_file = input_file.replace("-raw-dataset.csv", "-x-balanced-"+sampler_name+".npz")
    y_balanced_file = input_file.replace("-raw-dataset.csv", "-y-balanced-"+sampler_name+".npz")

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

    print "number of samples in inital training period = %d" % x_initialTrain.shape[0]
    print "1st val period: (cross) =%d, (non-cross)=%d" % (x_val_cross[0].shape[0], x_val_non_cross[0].shape[0])
    print "2nd val period: (cross) =%d, (non-cross)=%d" % (x_val_cross[1].shape[0], x_val_non_cross[1].shape[0])
    print "3rd val period: (cross) =%d, (non-cross)=%d" % (x_val_cross[2].shape[0], x_val_non_cross[2].shape[0])
    print "4th val period: (cross) =%d, (non-cross)=%d" % (x_val_cross[3].shape[0], x_val_non_cross[3].shape[0])
    print "5th val period: (cross) =%d, (non-cross)=%d" % (x_val_cross[4].shape[0], x_val_non_cross[4].shape[0])


    # do the 5-fold cross validation on a rolling basis
    aver_active_dis = 0
    sampler = supportSamplers[sampler_name]
    x_train = []
    y_train = []
    for i in range(5):
        # analyze data distribution
        print "The %dth validation" % i
        print "Distribution of active and inactive samples"
        dis = np.bincount(y_initialTrain)
        print dis
        aver_active_dis += dis[0]/(dis[0]+dis[1]+0.0)
        # deal with imbalanced dataset
        x_Train, y_Train = sampler.fit_resample(x_initialTrain, y_initialTrain)
        print "After dealing with imbalanced dataset"
        dis = np.bincount(y_Train)
        print dis
        x_train.append(x_Train)
        y_train.append(y_Train)
        # update the training set, merge the x_val_cross[i] and x_val_non_cross[i] with x_initialTrain
        x_initialTrain = np.concatenate((x_initialTrain, x_val_cross[i]), axis=0)
        y_initialTrain = np.concatenate((y_initialTrain, y_val_cross[i]), axis=0)
        x_initialTrain = np.concatenate((x_initialTrain, x_val_non_cross[i]), axis=0)
        y_initialTrain = np.concatenate((y_initialTrain, y_val_non_cross[i]), axis=0)
    # save the undersampled dataset
    np.savez(x_balanced_file, x_train[0], x_train[1], x_train[2], x_train[3], x_train[4])
    np.savez(y_balanced_file, y_train[0], y_train[1], y_train[2], y_train[3], y_train[4])

if __name__ == "__main__":
    main(sys.argv[1:])
