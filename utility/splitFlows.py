'''
This script is used to split tcp/udp flows from packet trace. A splited flow is represented by {srcAddr, srcPort, dstAddr, dstPort, protocol, [t1, t2, ..., tn], [p1, p2, ..., pn]} where ti and pi is the ith pkt arrival time and pkt length
'''
import pandas as pd
import numpy as np
import os, sys, getopt


def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:s:o:",["ifile=", 'statsfile',"ofile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -s <statsfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-s", "--statsfile"):
            stat_file = arg
        elif opt in ("-o", "--ofile"):
            outputFile = arg

    class flowEntry:
        def __init__(self,srcAddr,srcPort,dstAddr,dstPort,protocol, start, startLen):
            self.srcAddr = srcAddr
            self.srcPort = srcPort
            self.dstAddr = dstAddr
            self.dstPort = dstPort
            self.protocol = protocol
            self.arrivals = [start]
            self.pktLens = [startLen]

    class flowEntrySim:
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
        v_flows[flowID] = flowEntrySim(entry['Packets'],entry['Start'], entry['End'])
    print 'finish reading stats file'

    with open(outputFile,'w') as f:
        f.write('srcAddr,srcPort,dstAddr,dstPort,Protocol,arrivals,pktLens\n')

    flowTable = {}
    count = 0
    outputEntries = []
    for chunk in pd.read_csv(input_file, usecols=['Time','Source','Destination','Protocol','SrcPort','DesPort', 'Length'], chunksize=100000):
        print "Processing the %d th chunk" % count
        count += 1
        for index, entry in chunk.iterrows():
            if type(entry['SrcPort']) is not str and type(entry['DesPort']) is not str and (np.isnan(entry['SrcPort']) or np.isnan(entry['DesPort'])):
                continue
            if entry['Protocol'] == 'IPX' or entry['Protocol'] == 'NBIPX':
                continue

            entry['SrcPort'] = str(int(entry['SrcPort']))
            entry['DesPort'] = str(int(entry['DesPort']))

            flowID = entry['Source']+"-"+entry['SrcPort']+'-'+entry['Destination']+'-'+entry['DesPort']+'-'+entry['Protocol']
            if flowID in flowTable:
                flowTable[flowID].arrivals.append(entry['Time'])
                flowTable[flowID].pktLens.append(entry['Length'])
                if len(flowTable[flowID].pktLens) == v_flows[flowID].numPkt:
                    outputEntries.append(flowTable[flowID])
                    del flowTable[flowID]
            else:
                flowTable[flowID] = flowEntry(entry['Source'],entry['SrcPort'],entry['Destination'],entry['DesPort'],entry['Protocol'],entry['Time'], entry['Length'])

        with open(outputFile,'ab') as f:
            for flow in outputEntries:
                f.write('%s,%s,%s,%s,%s,%s,%s\n' % (flow.srcAddr,flow.srcPort,flow.dstAddr,flow.dstPort,flow.protocol,flow.arrivals,flow.pktLens))
        outputEntries = []

if __name__ == "__main__":
    main(sys.argv[1:])
