'''
this script is to generate the distribution of pkt inter-arrival time of train non-cross flows, train-cross flows, test-non-cross flows, and test-cross flows
'''
import pandas as pd
import numpy as np
import os, sys, getopt


def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:",["ifile=", 'timeRange='])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <timeRange>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -r <timeRange>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-r", "--timeRange"):
            timeRange = int(arg)

    # a time splot = 1 ms
    train_non_cross = 100000*[0]
    train_cross = 100000*[0]
    test_non_cross = 100000*[0]
    test_cross = 100000*[0]

    count = 0
    for chunk in pd.read_csv(input_file, usecols=['arrivals'], chunksize=100):
        print "Processing the %d th chunk" % count
        count += 1
        for index, entry in chunk.iterrows():
            arrivals = entry['arrivals'][1:-1].split(' ')
            arrivals = [float(t) for t in arrivals]
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
