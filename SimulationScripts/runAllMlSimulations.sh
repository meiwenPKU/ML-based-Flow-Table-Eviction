for tableSize in 1024 2048 4096
do
  echo "processing tableSize=$tableSize"
  python mlSimulator.py -i /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001.csv -s /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001-uni-flow.csv -T $tableSize -m /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001-rf-uni-random-${tableSize:0:1}k-10pkt.sav -N 10 -l /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001-labelModel-uni-random-${tableSize:0:1}k-10pkt.sav -p 0.65 -v 10 -r 5000 > /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001-ml-uni-random-${tableSize:0:1}k-10pkt-test-v2.txt
done
