'''
This script is to generate training dataset and validation dataset. Different from genData-v2.py, this script defines a flow is inactive if the next packet will arrive after t_arrival95, where t_arrival95 is defined as the 95% value in the packet inter-arrival time distribution
'''

import pandas as pd
import numpy as np
import os, sys, getopt, csv


def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:a:",["ifile=", 'trainRange=', 'arrival95'])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <tainRange>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -r <timeRange>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-r", "--timeRange"):
            trainRange = int(arg)
        elif opt in ("-a", "--arrival95"):
            arrival95 = float(arg)

    N_last = 10
    #write the header for the output file
    v_feature = ['start', 'end', 'cur', 't_last_pkt','Protocol','isEnd']
    for i in range(0,N_last):
        v_feature.append('p'+str(i)+'-len')
    for i in range(0,N_last-1):
        v_feature.append('i'+str(i)+'-interval')

    output_file = input_file.replace('-split-flows.csv', '-raw-dataset-arrival95.csv')

    with open(output_file,'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=v_feature)
        writer.writeheader()
    count = 0
    dataset = []
    for chunk in pd.read_csv(input_file, usecols=['Protocol','arrivals', 'pktLens'], chunksize=1000):
        print "Processing the %d th chunk" % count
        count += 1
        for index, entry in chunk.iterrows():
            arrivals = entry['arrivals'][1:-1].split(' ')
            arrivals = [float(t) for t in arrivals]
            # if the pkts arrival beyond the training range, visit the next flow
            if arrivals[0] >= trainRange:
                continue
            pktLens = entry['pktLens'][1:-1].split(' ')
            pktLens = [int(num) for num in pktLens]
            intervals = [arrivals[i+1]-arrivals[i] for i in range(len(arrivals)-1)]
            # generate data samples for the end of the flow
            if arrivals[-1] <= trainRange:
                # t_idle can be any value and the label is 1
                meanInterval = sum(intervals)/len(intervals)
                sample = [arrivals[0], arrivals[-1], arrivals[-1], 0, entry['Protocol'], 1]
                for j in range(0,N_last):
                    index = -(N_last-j)
                    if -index > len(pktLens):
                        sample.append(-1)
                    else:
                        sample.append(pktLens[index])
                for j in range(0,N_last-1):
                    index = 1-(N_last-j)
                    if -index > len(intervals):
                        sample.append(-1)
                    else:
                        sample.append(intervals[index])
                for i in range(10):
                    sample[3] = meanInterval*(1<<i)
                    dataset.append(sample[:])

            # generate data samples for internal flows
            #low = max(1, len(pktLens)-N_last+1)
            for i in range(-2, -len(pktLens)-1, -1):
                if arrivals[i] >= trainRange:
                    continue
                # generate one sample
                sample = [arrivals[0], arrivals[-1], arrivals[i], intervals[i+1],entry['Protocol'],0]
                for j in range(0,N_last):
                    index = i+1-(N_last-j)
                    if -index > len(pktLens):
                        sample.append(-1)
                    else:
                        sample.append(pktLens[index])
                for j in range(0,N_last-1):
                    index = i+2-(N_last-j)
                    if -index > len(intervals):
                        sample.append(-1)
                    else:
                        sample.append(intervals[index])
                dataset.append(sample[:])

        # write samples to csv files
        with open(output_file,'ab') as csvfile:
            csvwriter = csv.writer(csvfile)
            for sample in dataset:
                csvwriter.writerow(sample)
        # clear the sets
        dataset = []

if __name__ == "__main__":
    main(sys.argv[1:])
