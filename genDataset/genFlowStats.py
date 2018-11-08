'''
this script is to generate flow statistics (number of pkts, start time of the flow, and end time of the flow)
from packet trace. Here a flow is defined as <src IP, src port, dst IP, dst port, protocol>
'''
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
input_file = parent_path+"/unibs/unibs20090930.csv"
outputFile = parent_path + '/unibs/unibs20090930-uni-flow.csv'

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

data_raw = pd.read_csv(input_file, usecols=['Time','Source','Destination','Protocol','SrcPort','DesPort'])
#data_raw = data_raw.query('Time < 1')
print data_raw.shape
data_raw = data_raw.query('Protocol != "IPX" & Protocol != "NBIPX"')
print data_raw.shape
data_raw['Time'] = data_raw['Time'].astype(float)
data_raw = data_raw.sort_values(['Time'])

flowTable = {}
for index, entry in data_raw.iterrows():
    if index % 10000 == 0:
        print "processed %d packets" % index

    if type(entry['SrcPort']) is not str and type(entry['DesPort']) is not str and (np.isnan(entry['SrcPort']) or np.isnan(entry['DesPort'])):
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
