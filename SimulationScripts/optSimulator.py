'''
This script is used to simulate the scenario where the switch knows exactly which flow entries in its flow table are inactive,
and thus can replace a new comming flow with the inactive ones. If all flow entries in the flow table are active, evict the LRU entry.
'''
import pandas as pd
import numpy as np
import random
import csv
import os, sys, getopt

global numActiveFlow
def main(argv):
    global numActiveFlow
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:s:T:r:",["ifile=","statFile=","tableSize=","timeRange="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -s <statFile> -T <tableSize> -r <timeRange>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -s <statFile> -T <tableSize> -r <timeRange>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-s", "--statFile"):
            stat_file = arg
        elif opt in ("-T","--tableSize"):
            tableSize = int(arg)
        elif opt in ("-r", "--timeRange"):
            timeRange = int(arg)

    class flowEntry:
        def __init__(self,numPkt,start,end):
            self.numPkt = numPkt
            self.start = start
            self.duration = end - start
            self.arrived = 0

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


    class flowTableEntry:
        def __init__(self,lrt):
            self.lrt = lrt
            self.isActive = True

    numActiveFlow = 0
    flowTable = {}
    fullFlowTable = {}

    def removeLRU():
        global numActiveFlow
        min_lrt = 100000000
        for key,x in flowTable.items():
            if not x.isActive:
                min_key = key
                break
            if x.lrt < min_lrt:
                min_lrt = x.lrt
                min_key = key
        if flowTable[min_key].isActive:
            numActiveFlow -= 1
        del flowTable[min_key]

    numMissHit = 0
    numCapMiss = 0
    numSubFlow = 0

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
            if flowID in flowTable:
                # this is not a new flow
                flowTable[flowID].lrt = entry['Time']
            else:
                # this is a new flow
                numActiveFlow += 1
                if len(flowTable) == tableSize:
                    removeLRU()
                flowTable[flowID] = flowTableEntry(entry['Time'])
                numMissHit += 1
                if flowID in fullFlowTable:
                    numCapMiss += 1
                    fullFlowTable[flowID] += 1
                else:
                    fullFlowTable[flowID] = 0
                if numMissHit % 100 == 0:
                    print "TableSize=%d, numMissHit=%d, numCapMiss=%d, numActiveFlow=%d, time=%f" % (len(flowTable),numMissHit, numCapMiss, numActiveFlow, entry['Time'])

            v_flows[flowID].arrived += 1
            if flowTable[flowID].isActive:
                if v_flows[flowID].numPkt == v_flows[flowID].arrived:
                    flowTable[flowID].isActive = False
                    numActiveFlow -= 1
    print "numMissHit=%d" % numMissHit
    print "numFlow = %d" % len(fullFlowTable)
    print "numCapMiss = %d" % numCapMiss
    print "CapMiss distribution"
    for value in fullFlowTable.values():
        print value,

if __name__ == "__main__":
    main(sys.argv[1:])
