__author__ = 'niharika sharma'

"""
Implementation of the fast gradient algorithm to train the linear support vector machine with the squared hinge loss
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as mpl


def computegrad(x, y, beta, lam):
    """
    This function computes the gradient of objective function
    :param x:       array
                    Features
    :param y:       array
                    Labels
    :param beta:    array
                    coefficients
    :param lam:     float
                    regularization parameter
    :return:        a vector of size d
    """

    n = len(x)
    yx = y[:, np.newaxis] * x
    d = np.size(x, 1)
    loss_grad = np.zeros(d)
    for i in range(0, n):
        loss_grad += (-yx[i]) * max(0, 1 - yx[i].T.dot(beta))

    return 2 / n * loss_grad + lam * 2 * beta


def objective(x, y, beta, lam):
    """
    This function computes the objective function value
    :param x:       array
                    Features
    :param y:       array
                    Labels
    :param beta:    array
                    coefficients
    :param lam:     float
                    regularization parameter
    :return:        a value for the objective function
    """

    n = len(x)
    yx = y[:, np.newaxis] * x
    loss = 0
    for i in range(0, n):
        loss += max(0, 1 - yx[i].T.dot(beta)) ** 2

    return 1 / n * (loss) + lam * np.linalg.norm(beta) ** 2


def bt_line_search(beta, lam, step_size, alpha, gamma, max_iter, x, y):
    """
    This function computes the backtracking rule to find optimal step size for fast grad descent
    :param beta:    array
                    coefficients
    :param lam:     float
                    regularization parameter
    :param step_size: Step size for previous iteration
    :param alpha:   hyper-parameter for backtracking
    :param gamma:   hyper-parameter for backtracking
    :param max_iter: int
                    Maximum number of iterations
    :param x:       array
                    Features
    :param y:       array
                    Labels
    :return:        step size
    """

    compgrad_beta = computegrad(x, y, beta, lam)
    norm_compgrad_beta = np.linalg.norm(compgrad_beta)
    flag = 0
    iter = 0
    while flag == 0 and iter < max_iter:
        lhs = objective(x, y, beta - step_size * compgrad_beta, lam)
        rhs = objective(x, y, beta, lam) - alpha * step_size * norm_compgrad_beta ** 2
        if lhs < rhs:
            flag = 1
        else:
            step_size *= gamma
        iter += 1
    return step_size


def mylinearsvm(beta_init, theta_init, lam, step_size_init, alpha, gamma, max_iter, x, y):
    """
    This function performs the fast gradient descent
    :param beta_init:       array
                            coefficients | all equal to zeroes initially
    :param theta_init:      array
                            coefficients | all equal to zeroes initially
    :param lam:             float
                            regularization parameter
    :param step_size_init:  initial step size
    :param alpha:           hyper-parameter for backtracking
    :param gamma:           hyper-parameters for backtracking
    :param max_iter:        Maximum number of iterations (stopping criteria)
    :param x:               array
                            Features
    :param y:               array
                            Labels
    :return:                betas and thetas
    """

    beta = beta_init
    theta = theta_init
    betas = beta
    thetas = theta
    compgrad_theta = computegrad(x, y, theta, lam)
    iter = 0
    while iter < max_iter:
        # Find step size with backtracking rule
        step_size = bt_line_search(theta, lam, step_size_init, alpha, gamma, max_iter, x, y)
        # beta t+1
        beta_new = theta - step_size * compgrad_theta
        # theta t+1
        theta = beta_new + iter / (iter + 3) * (beta_new - beta)
        # store all the betas and thetas
        betas = np.vstack((betas, beta_new))
        thetas = np.vstack((thetas, theta))

        compgrad_theta = computegrad(x, y, theta, lam)
        beta = beta_new
        iter += 1
    return betas, thetas


def compute_misclassification_error(b, x, y):
    """
    This function computes the mis-classification error
    :param b: optimal coefficients
    :param x: features
    :param y: labels
    :return: misclassification error
    """
    y_pred = (np.dot(x, b) > 0) * 2 - 1
    return np.mean(y_pred != y)


def cross_validation(x_train, y_train, x_test, y_test, step_size_init, alpha, gamma, max_iter):
    """
    This function performs cross-validation
    :param x_train:         array
                            training features
    :param y_train:         array
                            training labels
    :param x_test:          array
                            training features
    :param y_test:          array
                            training labels
    :param step_size_init:  step size
    :param alpha:           hyper-parameter for backtracking
    :param gamma:           hyper-parameter for backtracking
    :param max_iter:        Maximum number of iterations (stopping criteria)
    :return:                Optimal value of lambda
    """

    d = np.size(x_train, 1)

    # create an array of lambda values
    lams = np.array([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 100, 1000])

    errors = []

    for lam in lams:
        beta_init = np.zeros(d)
        theta_init = np.zeros(d)
        betas_fastgrad, thetas_fastgrad = mylinearsvm(beta_init, theta_init, lam, step_size_init, alpha, gamma,
                                                      max_iter,
                                                      x_train, y_train)
        e = compute_misclassification_error(betas_fastgrad[-1, :], x_test, y_test)

        errors = np.append(errors, e)

    # visualize the process of cross-validation
    fig, ax = mpl.subplots()
    ax.plot(np.log(lams), errors, c='magenta', label=' Mis-classification error using fast gradient')
    ax.legend(loc='upper left')
    mpl.xlabel('log(lambda)')
    mpl.ylabel('Mis-classification error')
    mpl.show()
    return lams[np.argmin(errors)]


def initial_step_size(x_train, y_train, lam):
    """
    This function computes the initial step step
    :param x_train:     array
                        training features
    :param y_train:     array
                        training labels
    :param lam:         float
                        regularization parameter
    :return:            float
                        initial step size
    """

    d = np.size(x_train, 1)
    return 1 / (scipy.linalg.eigh(1 / len(y_train) * x_train.T.dot(x_train), eigvals=(d - 1, d - 1), eigvals_only=True)[
                    0] + lam)
