'''
this script is used to generate datasets for different N_last (number of packets for classification)
It takes the output from genData.py as the input, and output the dataset for different N_last
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
        opts, args = getopt.getopt(argv,"hi:o:N:",["ifile=","ofile=","Nlast="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile> -N <Nlast>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile> -N <Nlast>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in ("-N","--Nlast"):
            N_last = int(arg)

    #write the header for the output file
    v_feature = ['t_last_pkt','avg_lastN_pkt','std_lastN_pkt','Protocol','isEnd']
    for i in range(0,N_last):
        v_feature.append('p'+str(i)+'-len')

    with open(output_file,'w') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames=v_feature)
        writer.writeheader()

    # get the data
    data_raw = pd.read_csv(input_file)
    columns = data_raw.columns.values.tolist()
    with open(output_file,'ab') as csvfile:
        csvwriter = csv.writer(csvfile)
        for index, entry in data_raw.iterrows():
            if index % 10000 == 0:
                print 'processing %d packets' % index
            sample = [entry['t_last_pkt']]
            v_interval = [entry[label] for label in columns[-(N_last-1):] if entry[label]!=-1]
            if len(v_interval) != 0:
                sample.append(np.mean(v_interval))
                sample.append(np.std(v_interval))
            else:
                sample.append(0)
                sample.append(0)
            sample.append(entry['Protocol'])
            sample.append(entry['isEnd'])
            sample.extend([entry['p'+str(i)+'-len'] for i in range(10-N_last,10)])
            csvwriter.writerow(sample)


if __name__ == "__main__":
    main(sys.argv[1:])
