'''
this script is to generate flow statistics (number of pkts, start time of the flow, and end time of the flow)
from packet trace. Here a flow is defined as <src IP, src port, dst IP, dst port, protocol>
'''
import pandas as pd
import numpy as np
import os

#dir_path = os.path.dirname(os.path.realpath(__file__))
#parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
input_file = "/home/yang/sdn-flowTable-management/univ2/univ2.csv"
outputFile = '/home/yang/sdn-flowTable-management/univ2/univ2-uni-flow.csv'

class flowEntry:
    def __init__(self,srcAddr,srcPort,dstAddr,dstPort,protocol,start):
        self.srcAddr = srcAddr
        self.srcPort = srcPort
        self.dstAddr = dstAddr
        self.dstPort = dstPort
        self.protocol = protocol
        self.numPkt = 1
        self.start = start
        self.end = start

with open(outputFile,'w') as f:
    f.write('srcAddr,srcPort,dstAddr,dstPort,Packets,Protocol,Start,End\n')

flowTable = {}
count = 0
for chunk in pd.read_csv(input_file, usecols=['Time','Source','Destination','Protocol','SrcPort','DesPort'], chunksize=1000000):
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
            flowTable[flowID].numPkt += 1
            flowTable[flowID].end = entry['Time']
        else:
            flowTable[flowID] = flowEntry(entry['Source'],entry['SrcPort'],entry['Destination'],entry['DesPort'],entry['Protocol'],entry['Time'])

with open(outputFile,'ab') as f:
    for key,flow in flowTable.iteritems():
        f.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (flow.srcAddr,flow.srcPort,flow.dstAddr,flow.dstPort,flow.numPkt,flow.protocol,flow.start,flow.end))
