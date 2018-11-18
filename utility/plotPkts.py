'''
This script is to plot number of arrival pkts versus time
'''
import matplotlib.pyplot as plt
import numpy as np

# input0930 = "/home/yang/sdn-flowTable-management/unibs/unibs20090930-pkts.txt"
# input1001 = "/home/yang/sdn-flowTable-management/unibs/unibs20091001-pkts.txt"
# input1002 = "/home/yang/sdn-flowTable-management/unibs/unibs20091002-pkts.txt"
# granularity = 10
#
#
# def helper(input_file):
#     with open(input_file,'r') as f:
#         lines = f.readlines()
#         if len(lines) != 1:
#             print "wrong input file"
#         pkts = lines[0].split(',')
#         pkts = pkts[:-1]
#         pkts = [int(pkt)/float(granularity) for pkt in pkts]
#         # plot the pkts
#         time = range(0, len(pkts)*granularity, granularity)
#         time = np.array(time)
#         pkts = np.array(pkts)
#     return time, pkts
#
# time0930, pkts0930 = helper(input0930)
# time1001, pkts1001 = helper(input1001)
# time1002, pkts1002 = helper(input1002)
#
# plt.figure()
# plt.plot(time0930, pkts0930, 'r', label="UNIBS20090930")
# plt.plot(time1001, pkts1001, 'b', label="UNIBS20091001")
# plt.plot(time1002, pkts1002, 'g', label="UNIBS20091002")
#
# plt.xlabel('Time/s')
# plt.ylabel('Number of arrival packets per second')
# plt.legend()
# plt.show()


input1 = "/home/yang/sdn-flowTable-management/univ1/univ1-pkts.txt"
input2 = "/home/yang/sdn-flowTable-management/univ2/univ2-pkts.txt"

granularity = 1
def helper(input_file):
    with open(input_file,'r') as f:
        lines = f.readlines()
        if len(lines) != 1:
            print "wrong input file"
        pkts = lines[0].split(',')
        pkts = pkts[:-1]
        pkts = [int(pkt)/float(granularity) for pkt in pkts]
        # plot the pkts
        time = range(0, len(pkts)*granularity, granularity)
        time = np.array(time)
        pkts = np.array(pkts)
    return time, pkts

time1, pkts1 = helper(input1)
time2, pkts2 = helper(input2)

plt.figure()
plt.plot(time1, pkts1, 'r', label="UNIV1")
plt.plot(time2, pkts2, 'b', label="UNIV2")


plt.xlabel('Time/s')
plt.ylabel('Number of arrival packets per second')
plt.legend()
plt.show()
