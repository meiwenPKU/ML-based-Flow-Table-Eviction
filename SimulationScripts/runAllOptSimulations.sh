for tableSize in 1024 2048 4096
do
  echo "processing tableSize=$tableSize"
  python optSimulator.py -i /home/yang/sdn-flowTable-management/unibs/unibs20090930.csv -s /home/yang/sdn-flowTable-management/unibs/unibs20090930-uni-flow.csv -T $tableSize -r 10000 > /home/yang/sdn-flowTable-management/unibs/unibs20090930-opt-uni-$tableSize-test.txt
done
