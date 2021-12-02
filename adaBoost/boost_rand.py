#-------------------------------------------------------------------------
# AUTHOR: Anthony Spencer
# FILENAME: boost_rand.py
# SPECIFICATION: Ada Boost plus rand foreest
# FOR: CS 4210- project/adaboost
#-----------------------------------------------------------*/

from sklearn.ensemble import AdaBoostClassifier
import sklearn
import csv
from sklearn.ensemble import RandomForestClassifier 
X_test = []
Y_test = []
X_training = []
Y_training = []

h_acc=0
h_n=''
h_l=''



with open('heartTraining.csv', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        if i > 0:
            X_training.append(row[:-1])
            Y_training.append(row[-1])

#reading the data in a csv file
with open('heartTesting.csv', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        if i > 0:
            X_test.append(row[:-1])
            Y_test.append(row[-1])


n_estimators = [1,2,5,10,50,250,500,1000]
n_estimators2 = [1,2,5,10,50,250,500,1000]
l_rate=[0.01,0.1,.5,1]
depth=[1,2,3,4,5,10,15,20,30,50,100]
boot=[False,True]
for n in n_estimators:
    for l in l_rate:
        for k in n_estimators2:
            for j in depth:
                for b in boot:

                    model3 = RandomForestClassifier(n_estimators=k,max_depth=j,bootstrap=b)
                    
                    ab = AdaBoostClassifier(n_estimators=n,  learning_rate=l,base_estimator=model3)
                    model = ab.fit(X_training, Y_training)

                    y_pred = model.predict(X_test)


                    tempacc=sklearn.metrics.accuracy_score(Y_test, y_pred)

                    #print("for n = " + str(n) + " learning rate = "+ str(l))
                    #print("Accuracy:",tempacc)

                    if tempacc > h_acc:
                        print('new highest acc =' + str(tempacc) + ' at n = boost' + str(n) + " and learning rate = "+ str(l) + " tree estimators = " + str(k) + " depth =" + str(j) + " boot=" + str(b))
                        h_acc=tempacc
                        h_l=l
                        h_n=n


print('final  highest acc =' + str(h_acc) + ' at n = ' + str(h_n) + " and L = "+ str(h_l))