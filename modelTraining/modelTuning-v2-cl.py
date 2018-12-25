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

This script supports to pass a dictionary as command line arguments
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
from sklearn.linear_model import LogisticRegression

supportClfs = {'knn':    KNeighborsClassifier(),
"svm":    SVC(),
"dt":    DecisionTreeClassifier(max_depth=5),
"rf":    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
"nn":    MLPClassifier(alpha=1),
"ada":    AdaBoostClassifier(),
"gnb":    GaussianNB(),
"gbt":    GradientBoostingClassifier(),
"lr":  LogisticRegression() }

def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:N:c:p:",["ifile=", 'trainRange=', 'Npkt=', 'classifier', 'parameters'])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <tainRange> -N <Npkt> -c <classifier> -p <parameters>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -r <timeRange> -N <Npkt> -c <classifier: only be dt, rf, gnb, gbt, knn, svm, nn, ada> -p <parameters>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-r", "--timeRange"):
            trainRange = int(arg)
        elif opt in ("-N", "--Npkt"):
            Npkt = int(arg)
        elif opt in ('-c', "--classifier"):
            clf_name = arg
        elif opt in ("-p", "--parameters"):
            parameters = json.loads(arg)
    if clf_name not in supportClfs:
        print "%s is not supported, we only support decision tree, random forest, gaussian naive Bayes, gradient boosting tree, knn, svm, neural network, ada boosting" % clf
        exit()

    x_initialTrain_file = input_file.replace("-raw-dataset.csv", "-x-inital-train-"+str(Npkt)+"pkt.npz")
    y_initialTrain_file = input_file.replace("-raw-dataset.csv", "-y-inital-train-"+str(Npkt)+"pkt.npz")
    x_val_cross_file = input_file.replace("-raw-dataset.csv", "-x-val-cross-"+str(Npkt)+"pkt.npz")
    y_val_cross_file = input_file.replace("-raw-dataset.csv", "-y-val-cross-"+str(Npkt)+'.npz')
    x_val_non_cross_file = input_file.replace("-raw-dataset.csv", "-x-val-non-cross-"+str(Npkt)+"pkt.npz")
    y_val_non_cross_file = input_file.replace("-raw-dataset.csv", "-y-val-non-cross-"+str(Npkt)+"pkt.npz")

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

    train_conf = {}
    val_cross_conf = {}
    val_non_cross_conf = {}

    # do the 5-fold cross validation on a rolling basis
    aver_active_dis = 0
    clf = supportClfs[clf_name]
    N = [5]
    for i in range(N[0]):
        # analyze data distribution
        print "The %dth validation" % i
        print "Distribution of active and inactive samples"
        if y_initialTrain.shape[0] == 0:
            N[0] -= 1
            # update the training set, merge the x_val_cross[i] and x_val_non_cross[i] with x_initialTrain
            x_initialTrain = np.copy(x_val_cross[i])
            y_initialTrain = np.copy(y_val_cross[i])
            x_initialTrain = np.concatenate((x_initialTrain, x_val_non_cross[i]), axis=0)
            y_initialTrain = np.concatenate((y_initialTrain, y_val_non_cross[i]), axis=0)
            continue
        dis = np.bincount(y_initialTrain)
        print dis
        aver_active_dis += dis[0]/(dis[0]+dis[1]+0.0)
        # # normalize the data
        # scaler = StandardScaler()
        # x_initialTrain = scaler.fit_transform(x_initialTrain)
        # # protocol data should not be normalized
        # mean = scaler.mean_[3]
        # std = scaler.scale_[3]
        # x_initialTrain[:,3] = (x_initialTrain[:,3]*std+mean).astype(int)
        # x_initialTrain[:,3] = x_initialTrain[:,3]
        # # normalize the validation set
        # x_val_cross[i] = scaler.transform(x_val_cross[i])
        # x_val_cross[i][:,3] = (x_val_cross[i][:,3]*std+mean).astype(int)
        # x_val_non_cross[i] = scaler.transform(x_val_non_cross[i])
        # x_val_non_cross[i][:,3] = (x_val_non_cross[i][:,3]*std+mean).astype(int)
        #TODO: deal with imbalanced dataset
        print "================================================"
        print "preprocessing the classifier %s " % clf
        # set the parameters of the classifier
        setting_index = 0
        for setting in ParameterGrid(parameters):
            ts = time.gmtime()
            print (setting),
            print(time.strftime("%Y-%m-%d %H:%M:%S", ts))
            clf.set_params(**setting)
            # train the model, the metric is
            clf.fit(x_initialTrain, y_initialTrain)
            y_train_pred = clf.predict(x_initialTrain)
            conf_train = confusion_matrix(y_initialTrain, y_train_pred)
            if setting_index in train_conf:
                train_conf[setting_index].append(np.copy(conf_train))
            else:
                train_conf[setting_index] = [np.copy(conf_train)]
            print conf_train
            # evaluate the cross flows
            y_cross_predict = clf.predict(x_val_cross[i])
            conf_val_cross = confusion_matrix(y_val_cross[i], y_cross_predict)
            if setting_index in val_cross_conf:
                val_cross_conf[setting_index].append(np.copy(conf_val_cross))
            else:
                val_cross_conf[setting_index] = [np.copy(conf_val_cross)]
            #print "results for cross flows:"
            print conf_val_cross
            # evaluate the non-cross flows
            y_non_cross_predict = clf.predict(x_val_non_cross[i])
            conf_val_non_cross = confusion_matrix(y_val_non_cross[i], y_non_cross_predict)
            if setting_index in val_non_cross_conf:
                val_non_cross_conf[setting_index].append(np.copy(conf_val_non_cross))
            else:
                val_non_cross_conf[setting_index] = [np.copy(conf_val_non_cross)]
            #print "results for non-cross flows"
            print conf_val_non_cross
            setting_index += 1

        # update the training set, merge the x_val_cross[i] and x_val_non_cross[i] with x_initialTrain
        x_initialTrain = np.concatenate((x_initialTrain, x_val_cross[i]), axis=0)
        y_initialTrain = np.concatenate((y_initialTrain, y_val_cross[i]), axis=0)
        x_initialTrain = np.concatenate((x_initialTrain, x_val_non_cross[i]), axis=0)
        y_initialTrain = np.concatenate((y_initialTrain, y_val_non_cross[i]), axis=0)

    # print raw data
    print train_conf
    print val_cross_conf
    print val_non_cross_conf

    def stats(conf):
        train_active = [conf[i][0][0]/(conf[i][0][0] + conf[i][0][1]+0.0) for i in range(N[0])]
        train_inactive = [conf[i][1][1]/(conf[i][1][0] + conf[i][1][1]+0.0) for i in range(N[0])]
        precision = [0 if conf[i][1][1]==0 else conf[i][1][1]/(conf[i][1][1]+conf[i][0][1]+0.0) for i in range(N[0])]
        f1 = [0 if precision[i] == 0 else 2*precision[i]*train_inactive[i]/(precision[i]+train_inactive[i]) for i in range(N[0])]
        mean_active = np.mean(train_active)
        std_active = np.std(train_active)
        mean_inactive = np.mean(train_inactive)
        std_inactive = np.std(train_inactive)
        mean_precision = np.mean(precision)
        std_precision = np.std(precision)
        mean_f1 = np.mean(f1)
        std_f1 = np.std(f1)
        return mean_active, std_active, mean_inactive, std_inactive, mean_precision, std_precision, mean_f1, std_f1

    # print readable data
    print "The average active sample percent =%f" % (aver_active_dis/5.0)
    for key in train_conf:
        print ("{"),
        for param, v in ParameterGrid(parameters)[key].items():
            print ("%s:%s;"%(param, v)),
        print ("},"),
        #print (ParameterGrid(tuned_parameters)[key]),
        ma, sa, mia, sia, mp, sp, mf, sf = stats(train_conf[key])
        print ("%s, %s, %s, %s, %s, %s, %s, %s, " % (ma, sa, mia, sia, mp, sp, mf, sf)),
        ma, sa, mia, sia, mp, sp, mf, sf = stats(val_cross_conf[key])
        print ("%s, %s, %s, %s, %s, %s, %s, %s, " % (ma, sa, mia, sia, mp, sp, mf, sf)),
        ma, sa, mia, sia, mp, sp, mf, sf = stats(val_non_cross_conf[key])
        print ("%s, %s, %s, %s, %s, %s, %s, %s, " % (ma, sa, mia, sia, mp, sp, mf, sf)),
        val_conf = [val_cross_conf[key][i]+val_non_cross_conf[key][i] for i in range(N[0])]
        ma, sa, mia, sia, mp, sp, mf, sf = stats(val_conf)
        print ("%s, %s, %s, %s, %s, %s, %s, %s" % (ma, sa, mia, sia, mp, sp, mf, sf))

if __name__ == "__main__":
    main(sys.argv[1:])
