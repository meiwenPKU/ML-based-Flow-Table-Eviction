'''
This script is used to compute the packet inter-arrival time threshold to identify ON/OFF periods
We use the method in T. Benson etl, where a threshold arrival95 is defined as the 95% value in the packet inter-arrival time distribution. And the on time is the longest continual period during which all the packet inter-arrival times are smaller than arrival95. The off time is a period between two on periods.
'''
import numpy as np
input_train_cross = "/home/yang/sdn-flowTable-management/univ2/univ2-intervals-train-cross.txt"
input_train_non_cross = "/home/yang/sdn-flowTable-management/univ2/univ2-intervals-train-non-cross.txt"
input_test_cross = "/home/yang/sdn-flowTable-management/univ2/univ2-intervals-test-cross.txt"
input_test_non_cross = "/home/yang/sdn-flowTable-management/univ2/univ2-intervals-test-non-cross.txt"

cdf = 100000*[0]
total = [0]
def helper(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        regular = lines[0].split(',')[:-1]
        for i in range(len(regular)):
            cdf[i] += int(regular[i])
        regular = [int(num) for num in regular]
        nonRegular = lines[1].split(',')[:-1]
        total[0] += len(nonRegular)

helper(input_train_cross)
helper(input_train_non_cross)
helper(input_test_cross)
helper(input_test_non_cross)

total[0] += sum(cdf)
cumcdf = np.cumsum(cdf)
cdf = [num/float(total[0]) for num in cumcdf]
for i in range(100000):
    if cdf[i] >= 0.95:
        print "arrival95=%s" % i
        break
