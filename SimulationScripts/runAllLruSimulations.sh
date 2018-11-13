for tableSize in 1024 2048 4096
do
  echo "processing tableSize=$tableSize"
  python lruSimulator.py -i /home/yang/sdn-flowTable-management/unibs/unibs20091002.csv -s /home/yang/sdn-flowTable-management/unibs/unibs20091002-uni-flow.csv -T $tableSize -r 14000 > /home/yang/sdn-flowTable-management/unibs/unibs20091002-lru-uni-$tableSize-test.txt
done
