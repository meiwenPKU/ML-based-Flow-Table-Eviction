python genData-v3.py -i /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001.csv -o /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001-sim-random-1k-raw-dataset.csv -s 1024 -f /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001-uni-flow.csv -v 20 -r 5000 -p random

python genData-v3.py -i /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001.csv -o /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001-sim-random-2k-raw-dataset.csv -s 2048 -f /home/hemin/mountJared/sdn-flowTable-management/unibs/unibs20091001-uni-flow.csv -v 20 -r 5000 -p random

python genData-v3.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20091001.csv -o /home/hemin/sdn-flowTable-management/unibs/unibs20091001-sim-random-4k-raw-dataset.csv -s 4096 -f /home/hemin/sdn-flowTable-management/unibs/unibs20091001-uni-flow.csv -v 20 -r 5000 -p random


python splitData.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20091001-sim-random-1k-raw-dataset.csv -N 10 -r 5000

python splitData.py -i /home/yang/sdn-flowTable-management/unibs/unibs20091001-sim-random-2k-raw-dataset.csv -N 10 -r 5000

python splitData.py -i /home/yang/sdn-flowTable-management/unibs/unibs20091001-sim-random-4k-raw-dataset.csv -N 10 -r 5000
