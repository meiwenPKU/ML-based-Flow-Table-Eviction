for tableSize in 1024 2048 4096
do
  echo "processing tableSize=$tableSize"
  python optSimulator.py -i /home/yang/sdn-flowTable-management/univ1/univ1.csv -s /home/yang/sdn-flowTable-management/univ1/univ1-uni-flow.csv -T $tableSize -r 600 > /home/yang/sdn-flowTable-management/univ1/univ1-opt-uni-$tableSize-test.txt
done
