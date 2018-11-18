'''
This script is used to find ", " and replace it with " " such that it can be processed by pandas
'''

import fileinput

input = "/home/yang/sdn-flowTable-management/univ2/univ2-split-flows.csv"
count = 0
for line in fileinput.input(input, inplace = 1):
    count += 1
    print line.replace(", ", " ")
