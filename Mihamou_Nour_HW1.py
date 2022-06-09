import math
import numpy as np
import statistics as stats
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
 
 
"""
This document is a nearest-neighbor classifer/regressor that takes in any
size of dataset.
 
@author: Nour Mihamou
"""
idata = load_iris()
#Distance metrics
   # Calculates Manhattan distance
   # p = training data pt, q = test point
def M_dist(p, q):
   sum = 0
   for i in range(0,len(p)):
       sum += abs(p[i] - q[i])
   return sum
 
# Calculates Euclidean distance
def E_dist(p,q):
   sum = 0
   for i in range(0, len(p)):
       sum += (p[i] - q[i]) ** 2
   return math.sqrt(sum)
 
def my_sort(d):
 
   for i in range(len(d)):
       for j in range(len(d)-i-1):
           if d[j][-1] > d[j+1][-1]:
               d[j], d[j+1] = d[j+1], d[j]
   return d
  
#Nearest neighbor classifier/regressor
def run_kNN(data, targets, test_points, k, metric, regression):
   sorted_dist = []
   m_distances = []
   for testpt in range(0, len(test_points)):
       dist = 0.0
       for trainpt in range(0, len(data)):
           if metric == "euclidean":
               dist = (E_dist(data[trainpt], test_points[testpt]))
           else:
               dist = (M_dist(data[trainpt], test_points[testpt]))
           m_distances.append((targets[trainpt], dist))
   
   #distances sorted first k lowest to highest
   sorted_dist = my_sort(m_distances)
   
   kneigh = []
   for row in range(len(test_points)):
       kneigh.append(sorted_dist[row][0])
       
   #classifying only
   #   return the most common target based off the first k distances
   arr = []
   if regression == "False":
       mode1 = stats.mode(kneigh)
       if mode1 == "nan":
           mode1 = -1
       arr.append(mode1)
    
      
   # if regression = "True" then find the mean of the targets
   else:
       avg = stats.mean(kneigh)
       arr.append(avg)
       
   return kneigh
 
def run_tests():
   idata = load_iris()
 
   X = idata.data
   y = idata.target
   X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, test_size = 0.25, stratify=y, random_state=42)

  
   print("1. Classification (k = 3, metric = euclidean)")
   print("")
   myknn1 = run_kNN(X_train, y_train, X_test, 3, "euclidean", False)
   #Checking if myknn1 predicted y_test's
   count = 0
   myknn1_acc = 0.0
   for i in range(len(myknn1)):
       if myknn1[i] == y_test[i]:
           count += 1
   myknn1_acc = (count/len(y_test)) * 100
 
   print("Accuracy = " + str(myknn1_acc))
   print(" ")
  
   #Confusion Matrix
   con_mat = confusion_matrix(y_test, myknn1)
   print("Confusion Matrix:")
   print(con_mat)

   ####################################################################   
     
   print(" ")
   print(" ")
   print("2. Classification (k = 3, metric = manhattan)")
   myknn2 = run_kNN(X_train, y_train, X_test, 3, "manhattan", False)
   #Checking if myknn2 predicted y_test's
   count = 0
   myknn2_acc = 0.0
   for i in range(len(myknn2)):
       if myknn2[i] == y_test[i]:
           count += 1
   myknn2_acc = (count/len(y_test)) * 100
   print("Accuracy = " + str(myknn2_acc))
   print(" ")
  
   #Confusion Matrix
   con_mat = confusion_matrix(y_test, myknn2)
   print("Confusion Matrix:")
   print(con_mat)
  
   ####################################################################
  
   print(" ")
   print(" ")
   print("3. Classification (k = 5, metric = euclidean)")
   myknn3 = run_kNN(X_train, y_train, X_test, 5, "euclidean", False)
   #Checking if myknn3 predicted y_test's
   count = 0
   myknn3_acc = 0.0
   for i in range(len(myknn3)):
       if myknn3[i] == y_test[i]:
           count += 1
   myknn3_acc = (count/len(y_test)) * 100
   print("Accuracy = " + str(myknn3_acc))
   print("")
  
   #Confusion Matrix
   con_mat = confusion_matrix(y_test, myknn3)
   print("Confusion Matrix:")
   print(con_mat)
  
   ####################################################################
  
   print("")
   print("")
   print("4. Regression (k = 3, metric = euclidean)")
   myknn_regr1 = run_kNN(X_train, y_train, X_test, 3, "euclidean", True)
   #Checking if myknn predicted y_test's
   total = 0.0
   for i in range(len(myknn_regr1)):
       if myknn_regr1[i] == y_test[i]:
           total += (myknn_regr1[i] - y_test[i]) ** 2
   myknn_regr1_acc = total/len(y_test)
   print("Mean squared avg = " + str(myknn_regr1_acc))
  
   ####################################################################
  
   print("")
   print("")
   print("5. Regression (k = 3, metric = manhattan)")
   myknn_regr2 = run_kNN(X_train, y_train, X_test, 3, "manhattan", True)
   #Checking if myknn predicted y_test's
   total = 0.0
   for i in range(len(myknn_regr2)):
       if myknn_regr2[i] == y_test[i]:
           total += (myknn_regr2[i] - y_test[i]) ** 2
   myknn_regr2_acc = total/len(y_test)
   print("Mean squared avg = " + str(myknn_regr2_acc))
  
   ####################################################################
  
   print("")
   print("")
   print("6. Regression (k = 5, metric = euclidean)")
   myknn_regr3 = run_kNN(X_train, y_train, X_test, 5, "euclidean", True)
   #Checking if myknn predicted y_test's
   total = 0.0
   for i in range(len(myknn_regr3)):
       if myknn_regr3[i] == y_test[i]:
           total += (myknn_regr3[i] - y_test[i]) ** 2
   myknn_regr3_acc = total/len(y_test)
   print("Mean squared avg = " + str(myknn_regr3_acc))
  
   ####################################################################
  
   print("")
   print("")
   print("7. SKLearn Implementation (k = 3, metric = euclidean)")
   knn1 = KNeighborsClassifier(3, metric = 'euclidean')
   knn1.fit(X_train, y_train)
   knn1_pred = knn1.predict(X_test)
   knn1_acc = accuracy_score(knn1_pred, y_test) * 100
   print("SKLearn accuracy score:" + str(knn1_acc))
  
   ####################################################################
  
   print("")
   print("")
   print("8. SKLearn Implementation (k = 3, metric = manhattan)")
   knn2 = KNeighborsClassifier(3, metric = 'manhattan')
   knn2.fit(X_train, y_train)
   knn2_pred = knn2.predict(X_test)
   knn2_acc = accuracy_score(knn2_pred, y_test) * 100
   print("SKLearn accuracy score:" + str(knn2_acc))
  
   ####################################################################
  
   print("")
   print("")
   print("9. SKLearn Implementation (k = 5, metric = euclidean)")
   knn3 = KNeighborsClassifier(5, metric = 'euclidean')
   knn3.fit(X_train, y_train)
   knn3_pred = knn3.predict(X_test)
   knn3_acc = accuracy_score(knn3_pred, y_test) * 100
   print("SKLearn accuracy score:" + str(knn3_acc))