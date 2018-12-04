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
'''
import pandas as pd
import numpy as np
import time
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


# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     GradientBoostingClassifier()]
# names = ["Nearest Neighbors", "Linear SVM", "RBF/Poly SVM",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Gaussian Naive Bayes", "Gradient Boosting"]
#
# tuned_parameters=[{'n_neighbors':[3,5,10,20],'weights':['uniform','distance']},
#     {'kernel':['linear'],'C':[1e-2,1e-1,1e0,1e1],},
#     {'kernel':['rbf','poly'],'C':[1e-2,1e-1,1e0,1e1],'gamma':[1e-2,1e-1,1e0,1e1]},
#     {'criterion':['entropy'],'max_depth':[7,8,9,10,11,12,13]},
#     {'criterion':['entropy','gini'],'max_depth':[8, 9, 10,12,],'n_estimators':[20,30,40,50,60]},
#     {'hidden_layer_sizes':[{30},{15,15},{10,10,10}],'alpha':[1e-2,1e-1,1e0,1e1],'learning_rate': ['constant','invscaling', 'adaptive'],'learning_rate_init':[1e-2,1e-1,1e0,1e1]},
#     {'n_estimators':[90,110,130,150,170],'learning_rate':[0.8, 0.9,1.1,1.2]},
#     {'var_smoothing': [1e-9]},
#     {'n_estimators': [20, 30, 50, 80, 100], 'max_depth': [3,5,7,9,11], 'subsample': [0.4, 0.6, 0.8, 1.0], 'learning_rate': [0.01, 0.1, 0.5, 1]}]

classifiers = [ MLPClassifier(alpha=1)]
names = ["Neural network"]
tuned_parameters=[
    {'hidden_layer_sizes':[{30},{35},{40}],'alpha':[1e-3,1e-2],'learning_rate': ['adaptive'],'learning_rate_init':[1e-3,1e-2]}]

def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:N:",["ifile=", 'trainRange=', 'Npkt='])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <tainRange> -N <Npkt>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -r <timeRange>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-r", "--timeRange"):
            trainRange = int(arg)
        elif opt in ("-N", "--Npkt"):
            Npkt = int(arg)

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

    train_conf = [{} for i in range(len(names))]
    val_cross_conf = [{} for i in range(len(names))]
    val_non_cross_conf = [{} for i in range(len(names))]

    # do the 5-fold cross validation on a rolling basis
    aver_active_dis = 0
    for i in range(5):
        # analyze data distribution
        print "The %dth validation" % i
        print "Distribution of active and inactive samples"
        dis = np.bincount(y_initialTrain)
        print dis
        aver_active_dis += dis[0]/(dis[0]+dis[1]+0.0)
        # normalize the data
        scaler = StandardScaler()
        x_initialTrain = scaler.fit_transform(x_initialTrain)
        # protocol data should not be normalized
        mean = scaler.mean_[3]
        std = scaler.scale_[3]
        x_initialTrain[:,3] = (x_initialTrain[:,3]*std+mean).astype(int)
        # if np.bincount(x_initialTrain[:,3]).shape[0] != 2:
        #     print "Error happens when normalizing data"
        #     sys.exit()
        # normalize the validation set
        x_val_cross[i] = scaler.transform(x_val_cross[i])
        x_val_cross[i][:,3] = (x_val_cross[i][:,3]*std+mean).astype(int)
        x_val_non_cross[i] = scaler.transform(x_val_non_cross[i])
        x_val_non_cross[i][:,3] = (x_val_non_cross[i][:,3]*std+mean).astype(int)
        # if np.bincount(x_val_cross[i][:,3]).shape[0] != 2:
        #     print "Error happens when normalizing data"
        #     sys.exit()
        #TODO: deal with imbalanced dataset
        # for each model
        for index, clf in enumerate(classifiers):
            # set the parameters of the classifier
            print "================================================"
            print "preprocessing the classifier %s " % names[index]
            setting_index = 0
            for setting in ParameterGrid(tuned_parameters[index]):
                ts = time.gmtime()
                print (setting),
                print(time.strftime("%Y-%m-%d %H:%M:%S", ts))
                clf.set_params(**setting)
                # train the model, the metric is
                clf.fit(x_initialTrain, y_initialTrain)
                y_train_pred = clf.predict(x_initialTrain)
                conf_train = confusion_matrix(y_initialTrain, y_train_pred)
                if setting_index in train_conf[index]:
                    train_conf[index][setting_index].append(np.copy(conf_train))
                else:
                    train_conf[index][setting_index] = [np.copy(conf_train)]
                print conf_train
                # evaluate the cross flows
                y_cross_predict = clf.predict(x_val_cross[i])
                conf_val_cross = confusion_matrix(y_val_cross[i], y_cross_predict)
                if setting_index in val_cross_conf[index]:
                    val_cross_conf[index][setting_index].append(np.copy(conf_val_cross))
                else:
                    val_cross_conf[index][setting_index] = [np.copy(conf_val_cross)]
                #print "results for cross flows:"
                print conf_val_cross
                # evaluate the non-cross flows
                y_non_cross_predict = clf.predict(x_val_non_cross[i])
                conf_val_non_cross = confusion_matrix(y_val_non_cross[i], y_non_cross_predict)
                if setting_index in val_non_cross_conf[index]:
                    val_non_cross_conf[index][setting_index].append(np.copy(conf_val_non_cross))
                else:
                    val_non_cross_conf[index][setting_index] = [np.copy(conf_val_non_cross)]
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
        train_active = [conf[i][0][0]/(conf[i][0][0] + conf[i][0][1]+0.0) for i in range(5)]
        train_inactive = [conf[i][1][1]/(conf[i][1][0] + conf[i][1][1]+0.0) for i in range(5)]
        precision = [0 if conf[i][1][1]==0 else conf[i][1][1]/(conf[i][1][1]+conf[i][0][1]+0.0) for i in range(5)]
        f1 = [0 if precision[i] == 0 else 2*precision[i]*train_inactive[i]/(precision[i]+train_inactive[i]) for i in range(5)]
        mean_active = np.mean(train_active)
        std_active = np.std(train_active)
        mean_inactive = np.mean(train_inactive)
        std_inactive = np.std(train_inactive)
        mean_precision = np.mean(precision)
        std_precision = np.std(precision)
        mean_f1 = np.mean(f1)
        std_f1 = np.std(f1)
        return mean_active, std_active, mean_inactive, std_inactive, mean_precision, std_precision, mean_f1, std_f1

    # for index, clf in enumerate(names):
    #     print clf
    #     for key in train_conf[index]:
    #         print (ParameterGrid(tuned_parameters[index])[key]),
    #         print stats(train_conf[index][key]),
    #         print stats(val_cross_conf[index][key]),
    #         print stats(val_non_cross_conf[index][key]),
    #         val_conf = [val_cross_conf[index][key][i]+val_non_cross_conf[index][key][i] for i in range(5)]
    #         print stats(val_conf)

    # print readable data
    print "The average active sample percent =%f" % (aver_active_dis/5.0)
    for index, clf in enumerate(names):
        print clf
        for key in train_conf[index]:
            print ("{"),
            for param, v in ParameterGrid(tuned_parameters[index])[key].items():
                print ("%s:%s;"%(param, v)),
            print ("},"),
            #print (ParameterGrid(tuned_parameters)[key]),
            ma, sa, mia, sia, mp, sp, mf, sf = stats(train_conf[index][key])
            print ("%s, %s, %s, %s, %s, %s, %s, %s, " % (ma, sa, mia, sia, mp, sp, mf, sf)),
            ma, sa, mia, sia, mp, sp, mf, sf = stats(val_cross_conf[index][key])
            print ("%s, %s, %s, %s, %s, %s, %s, %s, " % (ma, sa, mia, sia, mp, sp, mf, sf)),
            ma, sa, mia, sia, mp, sp, mf, sf = stats(val_non_cross_conf[index][key])
            print ("%s, %s, %s, %s, %s, %s, %s, %s, " % (ma, sa, mia, sia, mp, sp, mf, sf)),
            val_conf = [val_cross_conf[index][key][i]+val_non_cross_conf[index][key][i] for i in range(5)]
            ma, sa, mia, sia, mp, sp, mf, sf = stats(val_conf)
            print ("%s, %s, %s, %s, %s, %s, %s, %s" % (ma, sa, mia, sia, mp, sp, mf, sf))

if __name__ == "__main__":
    main(sys.argv[1:])
