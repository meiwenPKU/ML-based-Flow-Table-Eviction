python genData-v3.py -i /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1.csv -o /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1-sim-1k-raw-dataset.csv -s 1024 -f /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1-uni-flow.csv -v 1 -r 600 -p lru

python genData-v3.py -i /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1.csv -o /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1-sim-2k-raw-dataset.csv -s 2048 -f /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1-uni-flow.csv -v 1 -r 600 -p lru

python genData-v3.py -i /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1.csv -o /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1-sim-4k-raw-dataset.csv -s 4096 -f /home/hemin/mountJared/sdn-flowTable-management/univ1/univ1-uni-flow.csv -v 1 -r 600 -p lru

python splitData.py -i /home/yang/sdn-flowTable-management/univ1/univ1-sim-random-1k-raw-dataset.csv -N 10 -r 600

python splitData.py -i /home/yang/sdn-flowTable-management/univ1/univ1-sim-random-2k-raw-dataset.csv -N 10 -r 600

python splitData.py -i /home/yang/sdn-flowTable-management/univ1/univ1-sim-random-4k-raw-dataset.csv -N 10 -r 600
