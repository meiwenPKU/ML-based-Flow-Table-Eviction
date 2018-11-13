#for tableSize in 1024 2048 4096
#do
#  echo "processing tableSize=$tableSize"
#  python lruSimulator.py -i /home/yang/sdn-flowTable-management/univ1/univ1.csv -s /home/yang/sdn-flowTable-management/univ1/univ1-uni-flow.csv -T $tableSize -r 600 > /home/yang/sdn-flowTable-management/univ1/univ1-lru-uni-$tableSize-test.txt
#done


for tableSize in 1024 2048 4096
do
  echo "processing tableSize=$tableSize"
  python mlSimulator.py -i /home/yang/sdn-flowTable-management/univ1/univ1.csv -s /home/yang/sdn-flowTable-management/univ1/univ1-uni-flow.csv -T $tableSize -m /home/yang/sdn-flowTable-management/univ1/rf-uni-random-${tableSize:0:1}k-10pkt.sav -N 10 -l /home/yang/sdn-flowTable-management/univ1/labelModel-uni-random-${tableSize:0:1}k-10pkt.sav -p 0.65 -v 1 -r 600 > /home/yang/sdn-flowTable-management/univ1/ml-uni-random-${tableSize:0:1}k-10pkt-test.txt
done
