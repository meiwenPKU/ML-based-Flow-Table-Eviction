#python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20090930-raw-dataset.csv -r 10000 -N 10 -c ada -p '{"n_estimators":[100,120],"learning_rate":[1.2]}'


python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20090930-raw-dataset.csv -r 10000 -N 10 -c knn -p '{"n_neighbors":[30,40,50],"weights":["uniform", "distance"]}'
