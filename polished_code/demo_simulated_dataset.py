__author__ = 'niharika sharma'

"""
This file runs the linear support vector machine with the squared hinge
loss on a simple simulated dataset. The function visualizes the
training process, and print the performance - the mis-classification error,
the accuracy score and the confusion matrix.
"""

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import preprocessing
import matplotlib.pyplot as mpl
import svm_squared_hinge as svm
from sklearn.metrics import accuracy_score, confusion_matrix

# Create Simulated Data
np.random.seed(0)
mu, sigma = 10, 100  # mean and standard deviation
class1 = pd.DataFrame(np.random.normal(mu, sigma, [500, 5]))
class1['y'] = 1
mu1, sigma1 = 100, 10  # mean and standard deviation
class2 = pd.DataFrame(np.random.normal(mu1, sigma1, [500, 5]))
class2['y'] = -1

# Join a sequence of arrays along an existing axis.
data = np.concatenate((class1, class2), axis=0)
y = data[:, 5]
x = data[:, 0:5]

# Divide the data into train, test sets
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data
std_scale = preprocessing.StandardScaler()
std_scale.fit(x_train)
x_train = std_scale.transform(x_train)
x_test = std_scale.transform(x_test)

# initialize hyper-parameters
d = np.size(x, 1)
max_iter = 100
alpha = 0.5
gamma = 0.8

# compute initial step size
step_size_init = svm.initial_step_size(x_train, y_train, lam=0.001)

# Run cross-validation to find the optimal value of lambda
lam = svm.cross_validation(x_train, y_train, x_test, y_test, step_size_init, alpha, gamma, max_iter)
beta_init = np.zeros(d)
theta_init = np.zeros(d)
step_size_init = svm.initial_step_size(x_train, y_train, lam)

# Run fast grad algo to find the beta coefficients
betas_fastgrad, thetas_fastgrad = svm.mylinearsvm(beta_init, theta_init, lam, step_size_init, alpha, gamma, max_iter,
                                                  x_train, y_train)

# we can use both betas_fastgrad, or thetas_fastgrad
gradf_vals = [svm.objective(x_train, y_train, betaf, lam) for betaf in betas_fastgrad]

# visualize the training process - Objective function value per iteration
mpl.plot(range(1, np.size(betas_fastgrad, 0) + 1), gradf_vals, "g")
mpl.show()

# Mis-classification error on test data
print('Mis-classification error on test data is ',
      svm.compute_misclassification_error(betas_fastgrad[-1, :], x_test, y_test))

# Predict the labels of test data using optimal beta values calculated using fast grad
my_y_pred = (np.dot(x_test, betas_fastgrad[-1, :]) > 0) * 2 - 1

# accuracy
print("Accuracy using my linear svm : {0:0.1f}%".format(accuracy_score(y_test, my_y_pred) * 100))

# Confusion Matrix
cm = confusion_matrix(y_test, my_y_pred)
print('Confusion Matrix using my linear svm :\n', cm)
