for inputfile in univ1/univ1
do
  echo "processing file=$inputfile"
  python splitFlows.py -i /home/yang/sdn-flowTable-management/${inputfile}.csv -s /home/yang/sdn-flowTable-management/${inputfile}-uni-flow.csv -o /home/yang/sdn-flowTable-management/${inputfile}-split-flows.csv
done
