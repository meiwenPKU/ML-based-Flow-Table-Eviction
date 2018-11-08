'''
This script is used to simulate the lru eviction strategy given a packet trace
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
        opts, args = getopt.getopt(argv,"hi:s:T:",["ifile=","statFile=","tableSize="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -s <statFile> -T <tableSize>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -s <statFile> -T <tableSize>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-s", "--statFile"):
            stat_file = arg
        elif opt in ("-T","--tableSize"):
            tableSize = int(arg)

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
    print 'finish reading stats file'


    class flowTableEntry:
        def __init__(self,lrt):
            self.lrt = lrt
            self.isActive = True
    #get the raw packets from traces
    data_raw = pd.read_csv(input_file, usecols=['Time','Source','Destination','Protocol','Length','SrcPort','DesPort'])
    data_raw = data_raw.query('Protocol == "TCP" | Protocol == "UDP"')

    print data_raw.shape
    data_raw['Time'] = data_raw['Time'].astype(float)
    print data_raw.shape
    data_raw = data_raw.sort_values(['Time'])
    print 'finish reading packet trace'

    numActiveFlow = 0
    flowTable = {}
    fullFlowTable = {}

    def removeLRU():
        global numActiveFlow
        min_lrt = 100000000
        for key,x in flowTable.items():
            if x.lrt < min_lrt:
                min_lrt = x.lrt
                min_key = key
        if flowTable[min_key].isActive:
            numActiveFlow -= 1
        del flowTable[min_key]

    numMissHit = 0
    numCapMiss = 0
    numSubFlow = 0

    for index, entry in data_raw.iterrows():
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
