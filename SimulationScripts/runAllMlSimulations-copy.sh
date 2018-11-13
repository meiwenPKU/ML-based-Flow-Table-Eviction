for tableSize in 2048 4096
do
  echo "processing tableSize=$tableSize"
  python mlSimulator.py -i /home/yang/sdn-flowTable-management/unibs/unibs20090930.csv -s /home/yang/sdn-flowTable-management/unibs/unibs20090930-uni-flow.csv -T ${tableSize} -m /home/yang/sdn-flowTable-management/unibs/unibs20090930-rf-uni-random-${tableSize:0:1}k-10pkt.sav -N 10 -l /home/yang/sdn-flowTable-management/unibs/unibs20090930-labelModel-uni-random-${tableSize:0:1}k-10pkt.sav -p 0.65 -v 20 -r 10000 > /home/yang/sdn-flowTable-management/unibs/unibs20090930-ml-uni-random-${tableSize:0:1}k-10pkt-test.txt

done
