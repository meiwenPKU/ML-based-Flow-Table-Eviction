'''
This script is used to simulate the scenario where ml model is applied to pick which flow entry should be evicted from the table.
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from datetime import datetime
from sklearn.metrics import recall_score
import os, sys, getopt

# the first element is for cross flows, and the second is for non-cross


def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:s:T:m:N:l:p:v:r:",["ifile=","statFile=","tableSize=","modelFile=","Nlast=","labelEncoder=","probThreshold=","interval=","timeRange"])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -s <statFile> -T <tableSize> -m <modelFile> -N <Nlast> -r <timeRange>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -s <statFile> -T <tableSize> -m <modelFile> -N <Nlast> -r <timeRange>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-s", "--statFile"):
            stat_file = arg
        elif opt in ("-T","--tableSize"):
            tableSize = int(arg)
        elif opt in ("-m", "--modelFile"):
            modelfile = arg
        elif opt in ("-N", "--Nlast"):
            N_last = int(arg)
        elif opt in ("-l", "--labelEncoder"):
            labelEncoder = arg
        elif opt in ("-p", "--probThreshold"):
            pe = float(arg)
        elif opt in ("-v", "--interval"):
            interval = int(arg)
        elif opt in ("-r", "--timeRange"):
            timeRange = int(arg)

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
            for i in range(0,N_last):
                if i >= N_last - len(entry.v_len):
                    sample.append(entry.v_len[i-N_last+len(entry.v_len)])
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
                    print "negative false prediction: %s, %f" % (', '.join(map(str, sample)), entry.prob_end)
            entry.lastUpdate = cur_time
            if entry.prob_end > 0.9:
                print "remove %r flow entry with id=%s, tLastVisit=%s, time=%s, confidence=%f" % (flowTable[key].isActive, key,entry.t_last_pkt, cur_time, entry.prob_end)
                if flowTable[key].isActive:
                    numActiveFlow[index] -= 1
                    print flowTable[key].__dict__
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

        print "remove %r flow entry with id=%s, tLastVisit=%s, time=%s, confidence=%f" % (flowTable[lru_key].isActive, lru_key,lru.t_last_pkt, cur_time, lru.prob_end)
        index = int(lru_key not in trainedFlows)
        if flowTable[lru_key].isActive:
            numActiveFlow[index] -= 1
            print flowTable[lru_key].__dict__
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
                    print "TableSize=%d, numMissHit=%d, numCapMissCross=%d, numCapMissNonCross=%d, numActiveFlowCross=%d, numActiveFlowNonCross=%d, numActivePredictCross=%d, numActivePredictNonCross=%d, numInactivePredictCross=%d, numInactivePredictNonCross=%d, numActiveCorrectPredictCross=%d, numActiveCorrectPredictNonCross=%d, numInactiveCorrectPredictCross=%d, numInactiveCorrectPredictNonCross=%d, time=%f" % (len(flowTable),numMissHit,numCapMiss[0],numCapMiss[1],numActiveFlow[0], numActiveFlow[1], numPredictActive[0], numPredictActive[1], numPredictInactive[0], numPredictInactive[1], numCorrectPredictActive[0], numCorrectPredictActive[1], numCorrectPredictInactive[0], numCorrectPredictInactive[1], entry['Time'])

                    numPredictActive = 2*[0]
                    numPredictInactive = 2*[0]
                    numCorrectPredictActive = 2*[0]
                    numCorrectPredictInactive = 2*[0]
            else:
                # this is not a new flow
                flowTable[flowID].v_interval.append(entry['Time']-flowTable[flowID].t_last_pkt)
                if len(flowTable[flowID].v_interval) > N_last-1:
                    flowTable[flowID].v_interval = flowTable[flowID].v_interval[1:]
                flowTable[flowID].t_last_pkt = entry['Time']
                flowTable[flowID].lastUpdate = 0
                flowTable[flowID].v_len.append(entry['Length'])
                if len(flowTable[flowID].v_len) > N_last:
                    flowTable[flowID].v_len = flowTable[flowID].v_len[1:]

            if flowTable[flowID].isActive:
                if flowTable[flowID].t_last_pkt >= v_flows[flowID].start + v_flows[flowID].duration:
                    flowTable[flowID].isActive = False
                    numActiveFlow[index] -= 1
    print "numMissHit=%d" % numMissHit
    print "numCapMissCross=%d" % numCapMiss[0]
    print "numCapMissNonCross=%d" % numCapMiss[1]
    print fullFlowTable

if __name__ == "__main__":
    main(sys.argv[1:])
