for tableSize in 1024 2048 4096
do
  echo "processing tableSize=$tableSize"
  python mlSimulator.py -i /home/yang/sdn-flowTable-management/univ2/univ2.csv -s /home/yang/sdn-flowTable-management/univ2/univ2-uni-flow.csv -T ${tableSize} -m /home/yang/sdn-flowTable-management/univ2/univ2-sim-random-${tableSize:0:1}k-rf-20t20d-10pkt.sav -N 10 -l /home/yang/sdn-flowTable-management/univ1/labelModel-uni-random-${tableSize:0:1}k-10pkt.sav -p 0.65 -v 1 -r 550 > /home/yang/sdn-flowTable-management/univ2/univ2-sim-random-${tableSize:0:1}k-10pkt-20t20d.txt
done
