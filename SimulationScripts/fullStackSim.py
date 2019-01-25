'''
This script is to implement full stack simulation, including model training and simulation
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
from sklearn.externals import joblib

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
        opts, args = getopt.getopt(argv, "hi:r:N:c:p:s:T:l:t:v:d:",["ifile=", 'timeRange=', 'Npkt=', 'classifier=', 'parameters=', 'statFile=', 'tableSize=', 'labelEncoder=', 'probThreshold=', 'interval=', 'dataset='])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <timeRange> -N <Npkt> -c <classifier> -p <parameters>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -r <timeRange> -N <Npkt> -c <classifier: only be dt, rf, gnb, gbt, knn, svm, nn, ada> -p <parameters>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-r", "--timeRange"):
            timeRange = int(arg)
        elif opt in ("-N", "--Npkt"):
            Npkt = int(arg)
        elif opt in ('-c', "--classifier"):
            clf_name = arg
        elif opt in ("-p", "--parameters"):
            parameters = json.loads(arg)
        elif opt in ("-s", "--statFile"):
            stat_file = arg
        elif opt in ("-T","--tableSize"):
            tableSize = int(arg)
        elif opt in ("-l", "--labelEncoder"):
            labelEncoder = arg
        elif opt in ("-t", "--probThreshold"):
            pe = float(arg)
        elif opt in ("-v", "--interval"):
            interval = int(arg)
        elif opt in ("-d", "--dataset"):
            dataset = arg

    if clf_name not in supportClfs:
        print "%s is not supported, we only support decision tree, random forest, gaussian naive Bayes, gradient boosting tree, knn, svm, neural network, ada boosting" % clf_name
        exit()

    x_initialTrain_file = dataset.replace("-raw-dataset.csv", "-x-inital-train-"+str(Npkt)+"pkt.npz")
    y_initialTrain_file = dataset.replace("-raw-dataset.csv", "-y-inital-train-"+str(Npkt)+"pkt.npz")
    x_val_cross_file = dataset.replace("-raw-dataset.csv", "-x-val-cross-"+str(Npkt)+"pkt.npz")
    y_val_cross_file = dataset.replace("-raw-dataset.csv", "-y-val-cross-"+str(Npkt)+'.npz')
    x_val_non_cross_file = dataset.replace("-raw-dataset.csv", "-x-val-non-cross-"+str(Npkt)+"pkt.npz")
    y_val_non_cross_file = dataset.replace("-raw-dataset.csv", "-y-val-non-cross-"+str(Npkt)+"pkt.npz")

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
        if y_initialTrain.shape[0] == 0:
            # update the training set, merge the x_val_cross[i] and x_val_non_cross[i] with x_initialTrain
            x_initialTrain = np.copy(x_val_cross[i])
            y_initialTrain = np.copy(y_val_cross[i])
        else:
            x_initialTrain = np.concatenate((x_initialTrain, x_val_cross[i]), axis=0)
            y_initialTrain = np.concatenate((y_initialTrain, y_val_cross[i]), axis=0)
        x_initialTrain = np.concatenate((x_initialTrain, x_val_non_cross[i]), axis=0)
        y_initialTrain = np.concatenate((y_initialTrain, y_val_non_cross[i]), axis=0)

    class flowEntry:
        def __init__(self,numPkt,start,end):
            self.numPkt = numPkt
            self.start = start
            self.duration = end - start
            self.arrived = 0

    class flowTableEntry:
        def __init__(self,length,t_last_pkt,protocol):
            self.t_last_pkt = t_last_pkt
            self.v_interval = []
            self.v_len = [length]
            self.prob_end = 0
            self.protocol = protocol
            self.lastUpdate = 0
            self.isActive = True

    # set the parameters of the classifier
    setting_index = 0
    clf = supportClfs[clf_name]
    for setting in ParameterGrid(parameters):
        ts = time.gmtime()
        print (setting),
        print(time.strftime("%Y-%m-%d %H:%M:%S", ts))
        name = clf_name
        for param, v in setting.items():
            name += "-" + param + "-" + str(v)
        simOutput = dataset.replace("-raw-dataset.csv", "-" + name + '-' + str(Npkt)+"pkt-simOutput.txt")
        simOutFile = open(simOutput, 'w')
        modelfile = dataset.replace("-raw-dataset.csv", "-" + name + '-' + str(Npkt)+"pkt.sav")
        if not os.path.isfile(modelfile) or True:
            clf.set_params(**setting)
            clf.fit(x_initialTrain, y_initialTrain)
            joblib.dump(clf,modelfile)

        numPredictInactive = 2*[0]
        numPredictActive = 2*[0]
        numCorrectPredictActive = 2*[0]
        numCorrectPredictInactive = 2*[0]
        numCapMiss = 2*[0]
        numActiveFlow = 2*[0]
        #get the flow statistics from stat file
        data_stat = pd.read_csv(stat_file)
        data_stat['srcPort'] = data_stat['srcPort'].astype(int)
        data_stat['dstPort'] = data_stat['dstPort'].astype(int)
        data_stat['Start'] = data_stat['Start'].astype(float)
        data_stat['End'] = data_stat['End'].astype(float)
        v_flows = {}
        trainedFlows = set()
        for index, entry in data_stat.iterrows():
            flowID = entry['srcAddr']+"-"+str(entry['srcPort'])+'-'+entry['dstAddr']+'-'+str(entry['dstPort'])+'-'+entry['Protocol']
            v_flows[flowID] = flowEntry(entry['Packets'],entry['Start'], entry['End'])
            if entry['Start'] < timeRange:
                trainedFlows.add(flowID)

        #load the scaler model and built RF model
        rf = joblib.load(modelfile)
        le = joblib.load(labelEncoder)
        protocols = le.classes_

        flowTable = {}
        fullFlowTable = {}

        def removeHPU(cur_time):
            for key, entry in flowTable.iteritems():
                if cur_time - entry.lastUpdate < interval and entry.lastUpdate != 0:
                    continue
                # get the feature vector of the entry
                sample = [cur_time - entry.t_last_pkt]
                if len(entry.v_interval) != 0:
                    sample.append(np.mean(entry.v_interval))
                    sample.append(np.std(entry.v_interval))
                else:
                    sample.append(0)
                    sample.append(0)
                sample.append((le.transform([entry.protocol]))[0])
                for i in range(0,Npkt):
                    if i >= Npkt - len(entry.v_len):
                        sample.append(entry.v_len[i-Npkt+len(entry.v_len)])
                    else:
                        sample.append(-1)
                # do the prediction
                entry.prob_end = rf.predict_proba(np.array(sample).reshape(1,-1))[0,1]
                # update the stats
                index = int(key not in trainedFlows)
                if entry.isActive:
                    numPredictActive[index] += 1
                    if entry.prob_end < 0.5:
                        numCorrectPredictActive[index] += 1
                else:
                    numPredictInactive[index] += 1
                    if entry.prob_end > 0.5:
                        numCorrectPredictInactive[index] += 1
                    else:
                        print >> simOutFile, "negative false prediction: %s, %f" % (', '.join(map(str, sample)), entry.prob_end)
                entry.lastUpdate = cur_time
                if entry.prob_end > 0.9:
                    print >> simOutFile, "remove %r flow entry with id=%s, tLastVisit=%s, time=%s, confidence=%f" % (flowTable[key].isActive, key,entry.t_last_pkt, cur_time, entry.prob_end)
                    if flowTable[key].isActive:
                        numActiveFlow[index] -= 1
                        print >> simOutFile, flowTable[key].__dict__
                    del flowTable[key]
                    return
            # get the flow entry with maximal prob_end
            lru = flowTable.values()[0]
            for key,x in flowTable.items():
                if x.prob_end >= lru.prob_end:
                    lru = x
                    lru_key = key
            if lru.prob_end < pe:
                lru = flowTable.values()[0]
                for key, x in flowTable.items():
                    if x.t_last_pkt <= lru.t_last_pkt:
                        lru = x
                        lru_key = key

            print >> simOutFile, "remove %r flow entry with id=%s, tLastVisit=%s, time=%s, confidence=%f" % (flowTable[lru_key].isActive, lru_key,lru.t_last_pkt, cur_time, lru.prob_end)
            index = int(lru_key not in trainedFlows)
            if flowTable[lru_key].isActive:
                numActiveFlow[index] -= 1
                print >> simOutFile, flowTable[lru_key].__dict__
            del flowTable[lru_key]

        numMissHit = 0
        # read the raw data from traces chunk by chunk
        for chunk in pd.read_csv(input_file, usecols=['Time','Source','Destination','Protocol','Length','SrcPort','DesPort'], chunksize=1000000):
            for index, entry in chunk.iterrows():
                if entry['Time'] <= timeRange or (entry['Protocol'] != 'TCP' and entry['Protocol'] != 'UDP'):
                    continue
                if type(entry['SrcPort']) is not str and type(entry['DesPort']) is not str and (np.isnan(entry['SrcPort']) or np.isnan(entry['DesPort'])):
                    continue
                entry['SrcPort'] = str(int(entry['SrcPort']))
                entry['DesPort'] = str(int(entry['DesPort']))
                flowID = entry['Source']+"-"+entry['SrcPort']+'-'+entry['Destination']+'-'+entry['DesPort']+'-'+entry['Protocol']
                v_flows[flowID].arrived += 1
                index = int(flowID not in trainedFlows)
                if flowID not in flowTable:
                    #this is a new flow
                    numActiveFlow[index] += 1
                    if len(flowTable) == tableSize:
                        removeHPU(entry['Time'])
                    flowTable[flowID] = flowTableEntry(entry['Length'],entry['Time'],entry['Protocol'])
                    numMissHit +=1
                    if flowID in fullFlowTable:
                        numCapMiss[index] += 1
                        fullFlowTable[flowID] += 1
                    else:
                        fullFlowTable[flowID] = 0

                    if numMissHit % 100 == 0:
                        print >> simOutFile, "TableSize=%d, numMissHit=%d, numCapMissCross=%d, numCapMissNonCross=%d, numActiveFlowCross=%d, numActiveFlowNonCross=%d, numActivePredictCross=%d, numActivePredictNonCross=%d, numInactivePredictCross=%d, numInactivePredictNonCross=%d, numActiveCorrectPredictCross=%d, numActiveCorrectPredictNonCross=%d, numInactiveCorrectPredictCross=%d, numInactiveCorrectPredictNonCross=%d, time=%f" % (len(flowTable),numMissHit,numCapMiss[0],numCapMiss[1],numActiveFlow[0], numActiveFlow[1], numPredictActive[0], numPredictActive[1], numPredictInactive[0], numPredictInactive[1], numCorrectPredictActive[0], numCorrectPredictActive[1], numCorrectPredictInactive[0], numCorrectPredictInactive[1], entry['Time'])

                        numPredictActive = 2*[0]
                        numPredictInactive = 2*[0]
                        numCorrectPredictActive = 2*[0]
                        numCorrectPredictInactive = 2*[0]
                else:
                    # this is not a new flow
                    flowTable[flowID].v_interval.append(entry['Time']-flowTable[flowID].t_last_pkt)
                    if len(flowTable[flowID].v_interval) > Npkt-1:
                        flowTable[flowID].v_interval = flowTable[flowID].v_interval[1:]
                    flowTable[flowID].t_last_pkt = entry['Time']
                    flowTable[flowID].lastUpdate = 0
                    flowTable[flowID].v_len.append(entry['Length'])
                    if len(flowTable[flowID].v_len) > Npkt:
                        flowTable[flowID].v_len = flowTable[flowID].v_len[1:]

                if flowTable[flowID].isActive:
                    if flowTable[flowID].t_last_pkt >= v_flows[flowID].start + v_flows[flowID].duration:
                        flowTable[flowID].isActive = False
                        numActiveFlow[index] -= 1
        print >> simOutFile, "numMissHit=%d" % numMissHit
        print >> simOutFile, "numCapMissCross=%d" % numCapMiss[0]
        print >> simOutFile, "numCapMissNonCross=%d" % numCapMiss[1]
        print >> simOutFile, fullFlowTable

if __name__ == "__main__":
    main(sys.argv[1:])
