'''
This script is used to find how many flows cross the training range and the testing range, i.e., the flow start at the time < timeRange and end at the time > timeRange
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

    pkts_train_non_cross = 0
    pkts_train_cross = 0
    pkts_test_non_cross = 0
    pkts_test_cross = 0
    flows_train_non_cross = 0
    flows_test_non_cross = 0
    flows_cross = 0
    count = 0
    for chunk in pd.read_csv(input_file, usecols=['arrivals'], chunksize=1000):
        print "Processing the %d th chunk" % count
        count += 1
        for index, entry in chunk.iterrows():
            arrivals = entry['arrivals'][1:-1].split(' ')
            arrivals = [float(t) for t in arrivals]
            if arrivals[0] <= timeRange:
                if arrivals[-1] <= timeRange:
                    # this is train non-cross flow
                    flows_train_non_cross += 1
                    pkts_train_non_cross += len(arrivals)
                else:
                    # this is cross flow
                    flows_cross += 1
                    for i in range(len(arrivals)-1):
                        index = int((arrivals[i+1]-arrivals[i])/0.001)
                        if arrivals[i+1] < timeRange:
                            pkts_train_cross += 1
                        else:
                            pkts_test_cross += 1
            else:
                # this is test non-cross flows
                flows_test_non_cross += 1
                pkts_test_non_cross += len(arrivals)
    print "trainFlows=%d, crossFlows=%d, testFlows=%d" % (flows_train_non_cross+flows_cross, flows_cross, flows_test_non_cross+flows_cross)
    print "pktsTrainNonCross=%d, pktsTrainCross=%d, pktsTestNonCross=%d, pktsTestCross=%d" % (pkts_train_non_cross, pkts_train_cross, pkts_test_non_cross, pkts_test_cross)

if __name__ == "__main__":
    main(sys.argv[1:])
