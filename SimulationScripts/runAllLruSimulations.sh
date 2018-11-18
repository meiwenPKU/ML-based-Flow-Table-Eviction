for tableSize in 1024 2048 4096
do
  echo "processing tableSize=$tableSize"
  python lruSimulator.py -i /home/yang/sdn-flowTable-management/SIMA2011/SIMA20111010.csv -s /home/yang/sdn-flowTable-management/SIMA2011/SIMA20111010-uni-flow.csv -T $tableSize -r 1800 > /home/yang/sdn-flowTable-management/SIMA2011/SIMA20111010-lru-uni-$tableSize-test.txt
done
