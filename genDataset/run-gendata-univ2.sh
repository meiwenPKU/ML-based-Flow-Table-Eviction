python genData-v3.py -i /home/hemin/sdn-flowTable-management/univ2/univ2.csv -o /home/hemin/sdn-flowTable-management/univ2/univ2-sim-random-1k-raw-dataset.csv -s 1024 -f /home/hemin/sdn-flowTable-management/univ2/univ2-uni-flow.csv -v 1 -r 550 -p random

python genData-v3.py -i /home/hemin/sdn-flowTable-management/univ2/univ2.csv -o /home/hemin/sdn-flowTable-management/univ2/univ2-sim-random-2k-raw-dataset.csv -s 2048 -f /home/hemin/sdn-flowTable-management/univ2/univ2-uni-flow.csv -v 1 -r 550 -p random

python genData-v3.py -i /home/hemin/sdn-flowTable-management/univ2/univ2.csv -o /home/hemin/sdn-flowTable-management/univ2/univ2-sim-random-4k-raw-dataset.csv -s 4096 -f /home/hemin/sdn-flowTable-management/univ2/univ2-uni-flow.csv -v 1 -r 550 -p random


python splitData.py -i /home/hemin/mountJared/sdn-flowTable-management/univ2/univ2-sim-random-1k-raw-dataset.csv -N 10 -r 550

python splitData.py -i /home/hemin/mountJared/sdn-flowTable-management/univ2/univ2-sim-random-2k-raw-dataset.csv -N 10 -r 550

python splitData.py -i /home/hemin/mountJared/sdn-flowTable-management/univ2/univ2-sim-random-4k-raw-dataset.csv -N 10 -r 550
