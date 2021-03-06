'''
This script is used to generate the training dataset for ML-based eviction.
A flow is defined by five-tuple: <src IP, src port, dst IP, dst port, protocol>
This script simulating the arriving of packets in the given packet trace,
and insert corresponding flow entries into the flow table. Once the flow table is
overflow, then applying the LRU/Random policy to remove the LRU/Random flow entry.
When the LRU policy/Random is applied, every flow entry in the flow table will generate one
data sample if it generates a data sample > interval second ago.
Every data sampel contains its feature vector and the label. And the feature vector
= <idle time, protocol, packet len of last 10 packets, inter-arrival time of last 10 packets>
'''
import pandas as pd
import numpy as np
import random
import csv
import os
import sys, getopt

def main(argv):
    input_file = ''
    output_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:s:f:v:r:p:",["ifile=","ofile=","tablesize=","Nlast=","statsfile=","interval=","timeRange=","policy"])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile> -s <tablesize> -f <statsfile> -v <interval> -r <timeRange> -p <policy>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile> -s <tablesize> -f <statsfile> -v <interval> -r <timeRange> -p <policy, can be either lru or random>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in ("-s","--tablesize"):
            tableSize = int(arg)
        elif opt in ("-f","--statsfile"):
            stat_file = arg
        elif opt in ("-v","--interval"):
            interval = int(arg)
        elif opt in ("-r","--timeRange"):
            time_range = int(arg)
        elif opt in ("-p", "--policy"):
            policy = arg

    N_last = 10
    class flowEntry:
        def __init__(self,numPkt,start,end):
            self.numPkt = numPkt
            self.start = start
            self.duration = end-start
            self.arrived = 0

    class flowFeature:
        def __init__(self,length,t_last_pkt,protocol):
            self.t_last_pkt = t_last_pkt
            self.t_lastN_pkt = 0
            self.v_interval = []
            self.v_len = [length]
            self.protocol = protocol
            self.isEnd = False
            self.isUpdate = True
            self.last_record = t_last_pkt

    #write the header for the output file
    v_feature = ['t_last_pkt','Protocol','isEnd']
    for i in range(0,N_last):
        v_feature.append('p'+str(i)+'-len')
    for i in range(0,N_last-1):
        v_feature.append('i'+str(i)+'-interval')

    with open(output_file,'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=v_feature)
        writer.writeheader()

    #get the flow statistics from stat file
    data_stat = pd.read_csv(stat_file)
    data_stat = data_stat.query('Start <' + str(time_range))
    data_stat['srcPort'] = data_stat['srcPort'].astype(int)
    data_stat['dstPort'] = data_stat['dstPort'].astype(int)
    data_stat['Start'] = data_stat['Start'].astype(float)
    data_stat['End'] = data_stat['End'].astype(float)
    v_flows = {}
    for index, entry in data_stat.iterrows():
        flowID = entry['srcAddr']+"-"+str(entry['srcPort'])+'-'+entry['dstAddr']+'-'+str(entry['dstPort'])+'-'+entry['Protocol']
        v_flows[flowID] = flowEntry(entry['Packets'],entry['Start'], entry['End'])

    #get the raw packets from traces
    data_raw = pd.read_csv(input_file, usecols=['Time','Source','Destination','Protocol','Length','SrcPort','DesPort'])
    data_raw = data_raw.query('Protocol == "TCP" | Protocol == "UDP"')
    print data_raw.shape
    data_raw['Time'] = data_raw['Time'].astype(float)
    data_raw = data_raw.query('Time < ' + str(time_range))
    print data_raw.shape
    data_raw = data_raw.sort_values(['Time'])

    flowTable = {}
    def eviction(cur_time):
        with open(output_file,'ab') as csvfile:
            csvwriter = csv.writer(csvfile)
            for key,x in flowTable.items():
                if not (x.isUpdate or cur_time-x.last_record > interval):
                    continue
                x.isUpdate = False
                x.last_record = cur_time
                sample = [cur_time -x.t_last_pkt,x.protocol]
                if x.isEnd:
                    sample.append(1)
                else:
                    sample.append(0)

                for i in range(0,N_last):
                    if i >= N_last-len(x.v_len):
                        sample.append(x.v_len[i-N_last+len(x.v_len)])
                    else:
                        sample.append(-1)
                for i in range(0,N_last-1):
                    if i >= N_last-1-len(x.v_interval):
                        sample.append(x.v_interval[i-N_last+1+len(x.v_interval)])
                    else:
                        sample.append(-1)
                csvwriter.writerow(sample)
        if policy == 'lru':
            min_lrt = flowTable.values()[0]

            for key,x in flowTable.items():
                if x.t_last_pkt <= min_lrt.t_last_pkt:
                    min_lrt = x
                    min_key = key
            del flowTable[min_key]
        else:
            key = random.choice(flowTable.keys())
            del flowTable[key]


    for index, entry in data_raw.iterrows():
        if index % 10000 == 0:
            print "processing %d th packets" % (index)
        if type(entry['SrcPort']) is not str and type(entry['DesPort']) is not str and (np.isnan(entry['SrcPort']) or np.isnan(entry['DesPort'])):
            continue

        entry['SrcPort'] = str(int(entry['SrcPort']))
        entry['DesPort'] = str(int(entry['DesPort']))
        flowID = entry['Source']+"-"+entry['SrcPort']+'-'+entry['Destination']+'-'+entry['DesPort']+'-'+entry['Protocol']
        if flowID in flowTable:
            # this is not a new flow
            flowTable[flowID].isUpdate = True
            if len(flowTable[flowID].v_interval) == N_last:
                flowTable[flowID].v_interval = flowTable[flowID].v_interval[1:]
                flowTable[flowID].v_interval.append(entry['Time']-flowTable[flowID].t_last_pkt)
            else:
                flowTable[flowID].v_interval.append(entry['Time']-flowTable[flowID].t_last_pkt)
            flowTable[flowID].t_last_pkt = entry['Time']
            flowTable[flowID].t_lastN_pkt = np.mean(flowTable[flowID].v_interval)
            flowTable[flowID].v_len.append(entry['Length'])
            if len(flowTable[flowID].v_len) > N_last:
                flowTable[flowID].v_len = flowTable[flowID].v_len[1:]
        else:
            # this is a new flow
            if len(flowTable) == tableSize:
                eviction(entry['Time'])
            flowTable[flowID] = flowFeature(entry['Length'], entry['Time'], entry['Protocol'])

        # find the label for this sample
        v_flows[flowID].arrived += 1
        if v_flows[flowID].numPkt == v_flows[flowID].arrived:
            flowTable[flowID].isEnd = True

if __name__ == "__main__":
    main(sys.argv[1:])
