# 4k
## decision Tree
echo "processing unibs20090930 4k decision tree"
python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20090930-sim-random-4k-raw-dataset.csv -r 10000 -N 10 -c dt -p '{"criterion":["entropy", "gini"],"max_depth":[10,20,30]}'
## ada boosting
echo "processing unibs20090930 4k ada boosting"
python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20090930-sim-random-4k-raw-dataset.csv -r 10000 -N 10 -c ada -p '{"n_estimators":[10,20,30],"learning_rate":[0.8, 0.9, 1.1]}'
## gradient boosting tree
echo "processing unibs20090930 4k gradient boosting tree"
python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20090930-sim-random-4k-raw-dataset.csv -r 10000 -N 10 -c gbt -p '{"n_estimators": [10,20,30], "max_depth": [10,20,30], "subsample": [0.6, 0.8, 1.0], "learning_rate": [0.01, 0.1, 0.5]}'
## logistic regression
echo "processing unibs20090930 4k logistic regression"
python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20090930-sim-random-4k-raw-dataset.csv -r 10000 -N 10 -c lr -p '{"C":[0.01, 0.1, 1.0],"penalty":["l1","l2"]}'
