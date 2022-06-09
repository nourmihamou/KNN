PROJECT TITLE: Nearest Neighbor Classifier/Regressor

PURPOSE OF PROJECT: This program is my own nearest-neighbor classifier/regressor. It allows datasets of varying sizes, uses any specified number of neighbors, gracefully handle ties, and permits/calculates different distance metrics (euclidean and Manhattan).

VERSION or DATE: April 13, 20222

AUTHORS: Nour Mihamou

USER INSTRUCTIONS: 
run_kNN(data, targets, test_points, k=1, metric='euclidean', regression=False)


data  is an N x d array of numeric values, where N is the total number of data points and d is the number of dimensions (i.e. how many coordinates each data point has). This is your training data - the points that count as potential neighbors.

targets  is a one-dimensional array of numeric values of length N representing the known ground-truth value for each point in data. Unless regression is True, these must be non-negative integers - you do not need to check for this, and your code's behavior is undefined in this case (e.g. I don't care how it reacts, including crashing). If regression is True, these may be any floating-point value.

test_points  is a T x d array of numeric values representing the points that you wish to run the kNN on. (T is the total number of test points, and can be any positive integer.)

k  is the number of neighbors to use in the kNN algorithm. It may be any positive integer <= N. (Again, you do not need to check this.)

metric  may be either euclidean or manhattan; use the specified distance metric when calculating nearest neighbors.

regression  determines whether you are performing classification or regression.

	When False, treat it as a classifier - the kNN algorithm should return the most common class among the k nearest neighbors. If there is a tie, the result should be the value -1, indicating that the algorithm was unable to choose which group to assign the test point to.

	When True, treat it as a regressor - the kNN algorithm should simply return the mean of the ground truth values for the k nearest neighbors.


Return value: The method should return a one-dimension array of size T, containing the answers for each of the test_points.


run_tests()

This is a method that takes zero arguments and runs all of the test cases you have tried, using train_test_split and/or cross-validation techniques. It should load appropriate datasets (the Iris dataset is a starting place) and execute run_knn method with various parameters, reporting the results - you can print a confusion matrix, report out TPR and FPR, and/or whatever else you think makes sense. You should also run the same data through the scikit-learn kNN classifier and compare the results.


