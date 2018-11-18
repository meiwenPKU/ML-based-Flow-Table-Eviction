'''
this script is to find the number of arrival pkts versus time
'''
import pandas as pd
import numpy as np
import os, sys, getopt


def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:d:g:",["ifile=", "ofile=", "duration", "granularity"])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile> -d <duration> -g <granularity>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile> -d <duration> -g <granularity>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            outputFile = arg
        elif opt in ("-d", "--duration"):
            duration = int(arg)
        elif opt in ("-g", "--granularity"):
            granularity = int(arg)

    Nt = duration / granularity
    numPkts = Nt * [0]

    flowTable = {}
    count = 0
    for chunk in pd.read_csv(input_file, usecols=['Time','Source','Destination','Protocol','SrcPort','DesPort'], chunksize=100000):
        print "Processing the %d th chunk" % count
        count += 1
        for index, entry in chunk.iterrows():
            if type(entry['SrcPort']) is not str and type(entry['DesPort']) is not str and (np.isnan(entry['SrcPort']) or np.isnan(entry['DesPort'])):
                continue
            if entry['Protocol'] == 'IPX' or entry['Protocol'] == 'NBIPX':
                continue
            numPkts[int(entry['Time']/granularity)] += 1

    with open(outputFile,'w') as f:
        for num in numPkts:
            f.write('%s,' % num)

if __name__ == "__main__":
    main(sys.argv[1:])
