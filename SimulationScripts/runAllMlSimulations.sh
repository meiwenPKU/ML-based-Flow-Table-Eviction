for tableSize in 1024
do
  echo "processing tableSize=$tableSize"
  python mlSimulator.py -i /home/yang/sdn-flowTable-management/unibs/unibs20091001.csv -s /home/yang/sdn-flowTable-management/unibs/unibs20091001-uni-flow.csv -T $tableSize -m /home/yang/sdn-flowTable-management/unibs/unibs20091001-sim-random-${tableSize:0:1}k-rf-10pkt.sav -N 10 -l /home/yang/sdn-flowTable-management/unibs/unibs20091001-labelModel-uni-random-${tableSize:0:1}k-10pkt.sav -p 0.65 -v 10 -r 5000 > /home/yang/sdn-flowTable-management/unibs/unibs20091001-sim-random-${tableSize:0:1}k-10pkt-10tree.txt
done
