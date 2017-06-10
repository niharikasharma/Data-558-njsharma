__author__ = 'niharika sharma'

"""
The file demonstrate an experimental comparison between your implementation and scikit-learn’s
on real-world dataset
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
import svm_squared_hinge as svm

# Load Data
spam_data = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
test_flag = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ',
                          header=None)
test_flag = np.array(test_flag).T[0]

x = np.asarray(spam_data)[:, 0:-1]
y = np.asarray(spam_data)[:, -1] * 2 - 1

# Divide the data into train, test sets
x_train = x[test_flag == 0, :]
x_test = x[test_flag == 1, :]
y_train = y[test_flag == 0]
y_test = y[test_flag == 1]

# Standardize the data.
std_scale = preprocessing.StandardScaler()
std_scale.fit(x_train)
x_train = std_scale.transform(x_train)
x_test = std_scale.transform(x_test)

# initialize hyper-parameters
d = np.size(x, 1)
max_iter = 109
alpha = 0.5
gamma = 0.8
lam = 0.001
beta_init = np.zeros(d)
theta_init = np.zeros(d)

# compute initial step size
step_size_init = svm.initial_step_size(x_train, y_train, lam)

# Run fast grad algo to find the beta coefficients
betas_fastgrad, thetas_fastgrad = svm.mylinearsvm(beta_init, theta_init, lam, step_size_init, alpha, gamma, max_iter,
                                                  x_train, y_train)

print('My fast grad algo Performance on spam data - ')
print()

# Predict the labels of test data using optimal beta values calculated using fast grad
my_y_pred = (np.dot(x_test, betas_fastgrad[-1, :]) > 0) * 2 - 1

# Mis-classification error on test data
print('Mis-classification error on test data using my linear svm is ',
      svm.compute_misclassification_error(betas_fastgrad[-1, :], x_test, y_test))

# accuracy
print("Accuracy using my linear svm : {0:0.1f}%".format(accuracy_score(y_test, my_y_pred) * 100))

# Confusion Matrix
cm = confusion_matrix(y_test, my_y_pred)
print('Confusion Matrix using my linear svm :\n', cm)

# Optimal coefficients value
print("Coefficient using my linear svm = \n", betas_fastgrad[-1, :])

print()
print()
print('Scikit Performance on spam data - ')
print()

# Scikit-Learn Linear SVM analysis
clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Mis-classification error on test data
print('Mis-classification error on test data using sckit-learn linear svm is ', (np.mean(y_pred != y_test)))

# accuracy
print("Accuracy using sckit-learn linear svm : {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('SKLearn Confusion Matrix:\n', cm)

# Optimal coefficients value
print("Coefficient scikit-learn linear svm = \n", clf.coef_)
