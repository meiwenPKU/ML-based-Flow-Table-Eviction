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

global numActiveFlow
global numPredictInactive
global numPredictActive
global numCorrectPredictInactive
global numCorrectPredictActive
global numCapMiss

def main(argv):
    global numActiveFlow
    global numPredictInactive
    global numPredictActive
    global numCorrectPredictInactive
    global numCorrectPredictActive
    global numCapMiss

    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:s:T:m:N:l:p:v:",["ifile=","statFile=","tableSize=","modelFile=","Nlast=","labelEncoder=","probThreshold=","interval="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -s <statFile> -T <tableSize> -m <modelFile> -N <Nlast>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -s <statFile> -T <tableSize> -m <modelFile> -N <Nlast>'
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
    class inactiveFlow:
        def __init__(self,start):
            self.start = start
            self.startProb = None
            self.end = None
            self.endProb = None

    #get the flow statistics from stat file
    data_stat = pd.read_csv(stat_file)
    data_stat['srcPort'] = data_stat['srcPort'].astype(int)
    data_stat['dstPort'] = data_stat['dstPort'].astype(int)
    data_stat['Start'] = data_stat['Start'].astype(float)
    data_stat['End'] = data_stat['End'].astype(float)
    v_flows = {}
    for index, entry in data_stat.iterrows():
        flowID = entry['srcAddr']+"-"+str(entry['srcPort'])+'-'+entry['dstAddr']+'-'+str(entry['dstPort'])+'-'+entry['Protocol']
        v_flows[flowID] = flowEntry(entry['Packets'],entry['Start'], entry['End'])

    #load the scaler model and built RF model
    rf = joblib.load(modelfile)
    le = joblib.load(labelEncoder)
    protocols = le.classes_

    #get the raw packets from traces
    data_raw = pd.read_csv(input_file, usecols=['Time','Source','Destination','Protocol','Length','SrcPort','DesPort'])
    data_raw = data_raw.loc[data_raw['Protocol'].isin(protocols)]
    print data_raw.shape
    data_raw['Time'] = data_raw['Time'].astype(float)
    #data_raw = data_raw.query('Time < ' + str(time_range))
    print data_raw.shape
    data_raw = data_raw.sort_values(['Time'])

    flowTable = {}
    inactiveFlows = {}
    numActiveFlow = 0
    numPredictActive = 0
    numPredictInactive = 0
    numCorrectPredictActive = 0
    numCorrectPredictInactive = 0
    def removeHPU(cur_time):
        global numActiveFlow
        global numPredictInactive
        global numPredictActive
        global numCorrectPredictInactive
        global numCorrectPredictActive
        global numCapMiss
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
            if key in inactiveFlows and not inactiveFlows[key].startProb:
                inactiveFlows[key].startProb = entry.prob_end
            if (entry.isActive):
                numPredictActive += 1
                if entry.prob_end < 0.5:
                    numCorrectPredictActive += 1
            else:
                numPredictInactive += 1
                if entry.prob_end > 0.5:
                    numCorrectPredictInactive += 1
                else:
                    print "negative false prediction: %s, %f" % (', '.join(map(str, sample)), entry.prob_end)
            entry.lastUpdate = cur_time
            if entry.prob_end > 0.9:
                print "remove %r flow entry with id=%s, tLastVisit=%s, time=%s, confidence=%f" % (flowTable[key].isActive, key,entry.t_last_pkt, cur_time, entry.prob_end)
                if flowTable[key].isActive:
                    numActiveFlow -= 1
                    numCapMiss += 1
                    print flowTable[key].__dict__
                else:
                    inactiveFlows[key].end = cur_time
                    inactiveFlows[key].endProb = entry.prob_end
                    print inactiveFlows[key].__dict__
                    del inactiveFlows[key]

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
        if flowTable[lru_key].isActive:
            numActiveFlow -= 1
            numCapMiss += 1
            print flowTable[lru_key].__dict__
        else:
            inactiveFlows[lru_key].end = cur_time
            inactiveFlows[lru_key].endProb = lru.prob_end
            print inactiveFlows[lru_key].__dict__
            del inactiveFlows[lru_key]
        del flowTable[lru_key]

    numMissHit = 0
    numCapMiss = 0
    for entry_index, entry in data_raw.iterrows():
        if type(entry['SrcPort']) is not str and type(entry['DesPort']) is not str and (np.isnan(entry['SrcPort']) or np.isnan(entry['DesPort'])):
            continue

        entry['SrcPort'] = str(int(entry['SrcPort']))
        entry['DesPort'] = str(int(entry['DesPort']))
        flowID = entry['Source']+"-"+entry['SrcPort']+'-'+entry['Destination']+'-'+entry['DesPort']+'-'+entry['Protocol']
        v_flows[flowID].arrived += 1
        if flowID not in flowTable:
            #this is a new flow
            numActiveFlow += 1
            if len(flowTable) == tableSize:
                removeHPU(entry['Time'])
            flowTable[flowID] = flowTableEntry(entry['Length'],entry['Time'],entry['Protocol'])
            numMissHit +=1
            if numMissHit % 100 == 0:
                print "TableSize=%d, numMissHit=%d, numCapMiss=%d, numActiveFlow=%d, Accuracy of active flow=%f, Accuracy of inactive flow=%f, time=%f" % (len(flowTable),numMissHit,numCapMiss,numActiveFlow, numCorrectPredictActive/(numPredictActive+0.00000001), numCorrectPredictInactive/(numPredictInactive+0.000000001),entry['Time'])
                numPredictActive = 0
                numPredictInactive = 0
                numCorrectPredictActive = 0
                numCorrectPredictInactive = 0
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
            if v_flows[flowID].numPkt == v_flows[flowID].arrived:
                flowTable[flowID].isActive = False
                inactiveFlows[flowID] = inactiveFlow(entry['Time'])
                numActiveFlow -= 1
    print "numMissHit=%d" % numMissHit
    print "numCapMiss=%d" % numCapMiss

if __name__ == "__main__":
    main(sys.argv[1:])
