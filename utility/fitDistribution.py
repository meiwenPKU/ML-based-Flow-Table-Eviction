'''
This script is used to fit the distribution of ontime, offtime, flow length in time, and flow inter-arrival time. In addition, we also caculate the mean data rate in the ON period and probability of tcp udp flows
'''
import pandas as pd
import numpy as np
import os, sys, getopt


def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:a:",["ifile=", 'timeRange=', 'arrival95='])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <timeRange>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -r <timeRange> -a <arrival95>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-r", "--timeRange"):
            timeRange = int(arg)
        elif opt in ("-a", "--arrival95"):
            arrival95 = int(arg) # the unit is in ms


    count = 0
    preArrival = 0
    onTimes = []
    offTimes = []

    for chunk in pd.read_csv(input_file, usecols=['Protocol','arrivals', 'pktLens'], chunksize=100):
        print "Processing the %d th chunk" % count
        count += 1
        for index, entry in chunk.iterrows():
            arrivals = entry['arrivals'][1:-1].split(' ')
            arrivals = [float(t) for t in arrivals]
            flowInterArrival = arrivals[0] - preArrival


            if arrivals[0] <= timeRange:
                if arrivals[-1] <= timeRange:
                    # this is non-cross flow
                    for i in range(len(arrivals)-1):
                        index = int((arrivals[i+1]-arrivals[i])/0.001)
                        if index >= 100000:
                            train_non_cross.append(index)
                        else:
                            train_non_cross[index] += 1
                else:
                    # this is cross flow
                    for i in range(len(arrivals)-1):
                        index = int((arrivals[i+1]-arrivals[i])/0.001)
                        if arrivals[i+1] < timeRange:
                            if index >= 100000:
                                train_cross.append(index)
                            else:
                                train_cross[index] += 1
                        else:
                            if index >= 100000:
                                test_cross.append(index)
                            else:
                                test_cross[index] += 1
            else:
                for i in range(len(arrivals)-1):
                    index = int((arrivals[i+1]-arrivals[i])/0.001)
                    if index > 100000:
                        test_non_cross.append(index)
                    else:
                        test_non_cross[index] += 1
    train_non_cross_file = input_file.replace('-split-flows.csv', '-intervals-train-non-cross.txt')
    train_cross_file = input_file.replace('-split-flows.csv', '-intervals-train-cross.txt')
    test_non_cross_file = input_file.replace('-split-flows.csv', '-intervals-test-non-cross.txt')
    test_cross_file = input_file.replace('-split-flows.csv', '-intervals-test-cross.txt')
    with open(train_cross_file,'w') as f:
        for index, num in enumerate(train_cross):
            if index == 100000:
                f.write('\n')
            f.write('%s,' % num)
    with open(train_non_cross_file,'w') as f:
        for index, num in enumerate(train_non_cross):
            if index == 100000:
                f.write('\n')
            f.write('%s,' % num)
    with open(test_cross_file,'w') as f:
        for index, num in enumerate(test_cross):
            if index == 100000:
                f.write('\n')
            f.write('%s,' % num)
    with open(test_non_cross_file,'w') as f:
        for index, num in enumerate(test_non_cross):
            if index == 100000:
                f.write('\n')
            f.write('%s,' % num)

if __name__ == "__main__":
    main(sys.argv[1:])
