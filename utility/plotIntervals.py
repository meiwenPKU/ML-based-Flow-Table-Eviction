'''
This script is to plot the distribution of per-packet inter-arrival time
'''
import matplotlib.pyplot as plt
import numpy as np

input_train_cross = "/home/yang/sdn-flowTable-management/unibs/unibs20091001-intervals-train-cross.txt"
input_train_non_cross = "/home/yang/sdn-flowTable-management/unibs/unibs20091001-intervals-train-non-cross.txt"
input_test_cross = "/home/yang/sdn-flowTable-management/unibs/unibs20091001-intervals-test-cross.txt"
input_test_non_cross = "/home/yang/sdn-flowTable-management/unibs/unibs20091001-intervals-test-non-cross.txt"

threshold = [1]
def helper(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        regular = lines[0].split(',')[:-1]
        regular = [int(num) for num in regular]
        nonRegular = lines[1].split(',')[:-1]
        nonRegular = [int(num) for num in nonRegular]
        total = sum(regular) + len(nonRegular)
        cumregular = np.cumsum(regular)
        cumregular = cumregular/float(total)
        cumregular = [value for value in cumregular if value <= threshold[0]]
    return np.arange(1,len(cumregular)+1), cumregular

interval_train_cross, cdf_train_cross = helper(input_train_cross)
interval_train_non_cross, cdf_train_non_cross = helper(input_train_non_cross)
interval_test_cross, cdf_test_cross = helper(input_test_cross)
interval_test_non_cross, cdf_test_non_cross = helper(input_test_non_cross)

plt.figure()
plt.semilogx(interval_train_cross, cdf_train_cross, 'r', label="Train cross flows")
plt.semilogx(interval_train_non_cross, cdf_train_non_cross, 'b', label="Train non-cross flows")
plt.semilogx(interval_test_cross, cdf_test_cross, 'g', label="Test cross flows")
plt.semilogx(interval_test_non_cross, cdf_test_non_cross, 'k', label = "Test non-cross flows")

plt.xlabel('Per packet inter-arrival time/ms')
plt.ylabel('CDF')
plt.legend()
plt.ylim(top = threshold[0])
plt.show()
