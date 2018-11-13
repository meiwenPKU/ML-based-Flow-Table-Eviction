'''
This script is used to collect the number of active flows in a pcap/csv file
Through the stats of active flows, we can judge whether a machine learning approach can achieve a good performance since the ML method works better in the case where some flow entries are not active.
'''

import pandas as pd
import sys

stat_file = "/home/yang/sdn-flowTable-management/unibs/unibs20091002-uni-flow.csv"
data_stat = pd.read_csv(stat_file)
data_stat['srcPort'] = data_stat['srcPort'].astype(int)
data_stat['dstPort'] = data_stat['dstPort'].astype(int)
data_stat['Start'] = data_stat['Start'].astype(float)
data_stat['End'] = data_stat['End'].astype(float)
data_stat = data_stat.sort_values(['Start'])


class flowTimeEntry:
    def __init__(self,start,duration):
        self.start = start
        self.duration = duration
        self.numActiveFlow = 0

v_flow_time = []
for index, entry in data_stat.iterrows():
    v_flow_time.append(flowTimeEntry(entry['Start'], entry['End']-entry['Start']))

numFlow = len(v_flow_time)


def findIndex(index,end):
    '''
    find the last flowTimeEntry satisfying self.start >= start and self.start <= end
    '''
    begIndex = index
    endIndex = numFlow - 1
    best_index = begIndex
    while endIndex >= begIndex:
        mid = (begIndex + endIndex) / 2
        if v_flow_time[mid].start > end:
            endIndex = mid - 1
        elif v_flow_time[mid].start < end:
            begIndex = mid + 1
        else:
            best_index = mid
            break
        if end - v_flow_time[mid].start > 0 and end-v_flow_time[mid].start < end-v_flow_time[best_index].start:
            best_index = mid
    return best_index

maxActiveFlow = 0
for index in range(0,numFlow):
    fin_index = findIndex(index,v_flow_time[index].start+v_flow_time[index].duration)
    for v_index in range(index,fin_index+1):
        v_flow_time[v_index].numActiveFlow += 1
    if (index+1) % 10 == 0:
        print "flowIndex=%d, time=%f, numActiveFlow=%d" % (index,v_flow_time[index].start,v_flow_time[index].numActiveFlow)
    if maxActiveFlow < v_flow_time[index].numActiveFlow:
        maxActiveFlow = v_flow_time[index].numActiveFlow
print "max num active flow is %d" % maxActiveFlow
