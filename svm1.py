import numpy as np
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import random

# Note: install scikit-learn and download Wholesale customers data.csv to the same directory as this script prior to running

f = "Wholesale customers data.csv"
data = pd.read_csv(f) # load the data into pandas data frame
data = data.astype(float)
# take as input two arrays: data1 holding the training samples, and an array data2 of class labels
data1 = data[:220]
data2 = data[220:]

#training samples, pattern matrix
X1 = data1.as_matrix(["Fresh","Detergents_Paper"])
Y1 = data1.as_matrix(["Channel"])
Y1 = Y1-1
Y1 = np.ravel(Y1) 

#print (len(X1))
#class labels, pattern matrix
X2 = data2.as_matrix(["Fresh","Detergents_Paper"])
Y2 = data2.as_matrix(["Channel"])
Y2 = np.ravel(Y2)
# changing indentity of digits
Y2 = Y2-1

# linear kernel computation
clf = svm.SVC(kernel='linear', C = 1.0)

clf.fit(X1,Y1)  
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)

# predict on training examples
output = clf.predict(X2)
#np.array([1]

w = clf.coef_[0]
print(w)

#fix the functional margin
a = -w[0] / w[1]

xx = np.linspace(0,70000)
yy = a * xx - clf.intercept_[0] / w[1]


#plt.figure()
h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

#gammaValue=0.01
#cValue=10
#classifier = svm.SVC(C = cValue, kernel='rbf', gamma = gammaValue, tol=0.1)
#classifier.fit(X1,Y1)
#predictionTest = classifier.predict(X2)
#effTest = 100*metrics.accuracy_score(Y2, predictionTest, normalize = True)
#print('training set off = ', effTest)


# color presentation of data 
#plt.figure()
ctr = 0
for c in range (0,len(output)):
    if output[c] == Y2[c]:
        ctr=ctr+1
    else:
        ctr=ctr
    if output[c] == Y2[c] and Y2[c] == 0:
        col = 'g'
    elif output[c] == Y2[c] and Y2[c] == 1:
        col = 'b'
    else:
        col = 'r'
    if Y2[c] == 0:
        mrk = 'o'
    else:
        mrk = 'x'
    plt.scatter(X2[c, 0], X2[c, 1], c = col, marker = mrk)
print(ctr/(c+1))

plt.ylabel('Detergent and Paper Sales')
plt.xlabel('Fresh Food Sales')
plt.title('SVM output')
plt.axis([0, 70000, 0, 40000])
#legend 
colors = ['g', 'r', 'b']
green_dot = plt.scatter(random(10), random(10), marker='o', color=colors[0])
red_dot = plt.scatter(random(10), random(10), marker='o', color=colors[1])
blue_cross = plt.scatter(random(10), random(10), marker='x', color=colors[2])
red_cross = plt.scatter(random(10), random(10), marker='x', color=colors[1])


plt.legend((green_dot, red_dot, blue_cross, red_cross ),
           ('Horeca True', 'Horeca False', 'Retail True', 'Retail False'),
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=8)
                        
plt.show()


