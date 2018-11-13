'''
This script is used to find how many flows cross the training range and the testing range, i.e., the flow start at the time < timeRange and end at the time > timeRange
'''
import pandas as pd
import numpy as np
import os

stat_file = "/home/yang/sdn-flowTable-management/unibs/unibs20091002-uni-flow.csv"
timeRange = 14000
numTestFlow = 0
numCrossFlow = 0
numTrainFlow = 0
#get the flow statistics from stat file
data_stat = pd.read_csv(stat_file)
data_stat['srcPort'] = data_stat['srcPort'].astype(int)
data_stat['dstPort'] = data_stat['dstPort'].astype(int)
data_stat['Start'] = data_stat['Start'].astype(float)
data_stat['End'] = data_stat['End'].astype(float)
for index, entry in data_stat.iterrows():
    flowID = entry['srcAddr']+"-"+str(entry['srcPort'])+'-'+entry['dstAddr']+'-'+str(entry['dstPort'])+'-'+entry['Protocol']
    if entry['Start'] <= timeRange:
        numTrainFlow += 1
    else:
        numTestFlow += 1
    if entry['Start'] <= timeRange and entry['End'] > timeRange:
        numCrossFlow += 1
print 'numTrainFlow=%d, numTestFlow=%d, numCrossFlow=%d' % (numTrainFlow, numTestFlow, numCrossFlow)
