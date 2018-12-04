python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/univ1/univ1-raw-dataset.csv -r 600 -N 10 -c ada -p '{"n_estimators":[50,60,70,80],"learning_rate":[0.5,0.6,0.7]}'

python modelTuning-v2-cl.py -i /home/hemin/sdn-flowTable-management/univ1/univ1-raw-dataset.csv -r 600 -N 10 -c knn -p '{"n_neighbors":[30,40,50],"weights":["uniform", "distance"]}'
