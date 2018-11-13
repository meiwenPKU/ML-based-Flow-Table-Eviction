'''
This script is used to convert a tshark generated csv file to the csv file used in our programs. It includes use different headers and also generating src port and dst port instead of tcp/udp src/dst port. In this conversion, only two protocols will be recognized: tcp and udp. Packets belonging to other protocols (e.g., icmp) will be dropped
'''
import pandas as pd
import numpy as np
import random
import csv
import os
import sys, getopt
import socket

protocols = {num:name[8:] for name,num in vars(socket).items() if name.startswith("IPPROTO")}

input_file = "/home/yang/sdn-flowTable-management/SIMA2011/SIMA2011-1010-ts.csv"
output_file = "/home/yang/sdn-flowTable-management/SIMA2011/SIMA2011-1010.csv"

#write the header for the output file
v_feature = ["Time","Source","Destination","Protocol","Length","Info","DesPort","SrcPort"]

with open(output_file,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames=v_feature)
    writer.writeheader()

with open(output_file,'ab') as csvfile:
    csvwriter = csv.writer(csvfile)
    # read data chunk by chunk to avoid memory issue
    for chunk in pd.read_csv(input_file, chunksize=100000):
        print "Processing a new chunk"
        for index, entry in chunk.iterrows():
            if not entry['ip.proto'] or (entry['ip.proto'] != 6 and entry['ip.proto'] != 17):
                continue
            sample = [entry['frame.time_relative'], entry['ip.src'], entry['ip.dst']]
            if entry['ip.proto'] == 6:
                sample.append('TCP')
                sample.append(entry['frame.len'])
                sample.append('')
                sample.append(int(entry['tcp.dstport']))
                sample.append(int(entry['tcp.srcport']))
            elif entry['ip.proto'] == 17:
                sample.append('UDP')
                sample.append(entry['frame.len'])
                sample.append('')
                sample.append(int(entry['udp.dstport']))
                sample.append(int(entry['udp.srcport']))
            else:
                continue
            csvwriter.writerow(sample)
