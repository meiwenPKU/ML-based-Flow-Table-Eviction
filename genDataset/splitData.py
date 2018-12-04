'''
This script is used to split data on a rolling basis which can enable cross validation on time seriless data set.
'''
import pandas as pd
import numpy as np
import os, sys, getopt, csv
def main(argv):
    input_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:r:N:",["ifile=", 'trainRange=', 'Npkt='])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -r <tainRange> -N <Npkt>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -r <timeRange>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-r", "--timeRange"):
            trainRange = int(arg)
        elif opt in ("-N", "--Npkt"):
            Npkt = int(arg)

    # TODO: now we only consider two protocols, need to consider all protocols in the future
    protocols = {'TCP':0, 'UDP':1}

    valT = trainRange / 10.0
    trainT = trainRange / 2.0
    x_initialTrain = []
    y_initialTrain = []
    x_val_cross = [[] for i in range(5)]
    y_val_cross = [[] for i in range(5)]
    x_val_non_cross = [[] for i in range(5)]
    y_val_non_cross = [[] for i in range(5)]

    # read the input csv file and get the whole dataset
    count = 0
    for chunk in pd.read_csv(input_file, chunksize=100000):
        print "Processing the %d th chunk" % count
        count += 1
        columns = chunk.columns.values.tolist()
        for index, entry in chunk.iterrows():
            # get the sample
            if entry['Protocol'] not in protocols:
                continue
            sample = [entry['t_last_pkt']]
            v_interval = [entry[label] for label in columns[-(Npkt-1):] if entry[label]!=-1]
            if len(v_interval) != 0:
                sample.append(np.mean(v_interval))
                sample.append(np.std(v_interval))
            else:
                sample.append(0)
                sample.append(0)
            sample.append(protocols[entry['Protocol']])
            sample.extend([entry['p'+str(i)+'-len'] for i in range(10-Npkt,10)])
            y = entry['isEnd']
            if entry['cur'] <= trainT:
                x_initialTrain.append(sample[:])
                y_initialTrain.append(y)
            else:
                index = int((entry['cur'] - trainT)/valT)
                if index == 5:
                    print entry['cur']
                    continue
                if entry['start'] < trainT + index * valT:
                    x_val_cross[index].append(sample[:])
                    y_val_cross[index].append(y)
                else:
                    x_val_non_cross[index].append(sample[:])
                    y_val_non_cross[index].append(y)
    # save the array into files
    x_initialTrain_file = input_file.replace("-raw-dataset.csv", "-x-inital-train-"+str(Npkt)+"pkt")
    y_initialTrain_file = input_file.replace("-raw-dataset.csv", "-y-inital-train-"+str(Npkt)+"pkt")
    x_val_cross_file = input_file.replace("-raw-dataset.csv", "-x-val-cross-"+str(Npkt)+"pkt")
    y_val_cross_file = input_file.replace("-raw-dataset.csv", "-y-val-cross-"+str(Npkt))
    x_val_non_cross_file = input_file.replace("-raw-dataset.csv", "-x-val-non-cross-"+str(Npkt)+"pkt")
    y_val_non_cross_file = input_file.replace("-raw-dataset.csv", "-y-val-non-cross-"+str(Npkt)+"pkt")
    np.savez(x_initialTrain_file, x_initialTrain)
    np.savez(y_initialTrain_file, y_initialTrain)
    np.savez(x_val_cross_file, x_val_cross[0], x_val_cross[1], x_val_cross[2], x_val_cross[3], x_val_cross[4])
    np.savez(y_val_cross_file, y_val_cross[0], y_val_cross[1], y_val_cross[2], y_val_cross[3], y_val_cross[4])
    np.savez(x_val_non_cross_file, x_val_non_cross[0], x_val_non_cross[1], x_val_non_cross[2], x_val_non_cross[3], x_val_non_cross[4])
    np.savez(y_val_non_cross_file, y_val_non_cross[0], y_val_non_cross[1], y_val_non_cross[2], y_val_non_cross[3], y_val_non_cross[4])

if __name__ == "__main__":
    main(sys.argv[1:])
