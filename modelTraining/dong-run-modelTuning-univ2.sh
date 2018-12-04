# python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/univ2/univ2-raw-dataset.csv -r 550 -N 10 -c ada -p '{"n_estimators":[110,130,150],"learning_rate":[0.8, 0.9,1.1]}'
#
# python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/univ2/univ2-raw-dataset.csv -r 550 -N 10 -c dt -p '{"criterion":["entropy"],"max_depth":[9,10,11]}'

python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/univ2/univ2-raw-dataset.csv -r 550 -N 10 -c rf -p '{"criterion":["entropy","gini"],"max_depth":[8, 9, 10,12],"n_estimators":[30,40,50]}'

python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/univ2/univ2-raw-dataset.csv -r 550 -N 10 -c gnb -p '{"priors": [None]}'
