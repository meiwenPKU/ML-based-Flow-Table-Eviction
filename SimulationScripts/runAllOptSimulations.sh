for tableSize in 1024 2048 4096
do
  echo "processing tableSize=$tableSize"
  python optSimulator.py -i /home/yang/sdn-flowTable-management/univ2/univ2.csv -s /home/yang/sdn-flowTable-management/univ2/univ2-uni-flow.csv -T $tableSize -r 550 > /home/yang/sdn-flowTable-management/univ2/univ2-opt-uni-$tableSize-test.txt
done
