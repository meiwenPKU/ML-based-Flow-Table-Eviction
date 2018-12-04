python modelTuning-v2-cl.py -i /home/yang/sdn-flowTable-management/unibs/unibs20091001-raw-dataset.csv -r 5000 -N 10 -c rf -p '{"criterion":["gini"],"max_depth":[12, 15, 17, 20],"n_estimators":[40]}'

# python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/unibs/unibs20091001-raw-dataset.csv -r 5000 -N 10 -c knn -p '{"n_neighbors":[30,40,50],"weights":["uniform", "distance"]}'
