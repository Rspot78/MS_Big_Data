# -*- coding: utf-8 -*-
"""@authors: bellet, gramfort, salmon
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model

plt.close('all')


###############################################################################
#                Stochastic gradient
###############################################################################


def decision_function(X, w):
    """fonction de prediction a partir d'un vecteur directeur"""
    return np.dot(X, w[1:]) + w[0]


def predict(X, w):
    """fonction de prediction de classe a partir d'un vecteur directeur"""
    return np.sign(decision_function(X, w))


def stochastic_gradient(X, y, gamma, n_iter, w_ini, loss="mse",
                        alpha=0):
    """Stochastic gradient algorithm

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,)
        The targets.
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size (QUESTION 5).
    n_iter : int
        The number of iterations
    w_ini : array, shape (n_features + 1,)
        The initial value of w.
    loss : str
        The type of loss function to use, e.g. "hinge" ou "mse".
    alpha : float
        The regularization coefficient.
        QUESTION 3
    average : bool
        Do an averaged stochastic gradient.
        QUESTION 2

    Returns
    -------
    w : array, shape (n_features + 1,)
        The final weights.
    all_w : array, shape (n_iter, n_features + 1)
        The weights across iterations.
    pobj : array, shape (n_iter,)
        The evolution of the cost function across iterations.
    """
    n_samples = X.shape[0]
    X = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
    all_w = np.zeros((n_iter, w_ini.size))
    all_w[0] = w_ini
    w = w_ini.copy()
    w_avg = w_ini.copy()
    pobj = np.zeros(n_iter)
    pobj_avg = np.zeros(n_iter)
    t0 = n_samples   # average only the after one epoch
    if not callable(gamma):
        def gamma_func(t):
            return gamma
    else:
        gamma_func = gamma

    for t in range(n_iter):
        idx = np.random.randint(n_samples)
        if loss == "mse":
            pobj[t] = 0.5 * np.mean((y - np.dot(X, w)) ** 2)
            pobj_avg[t] = 0.5 * np.mean((y - np.dot(X, w_avg)) ** 2)
            if alpha > 0:
                pobj[t] += 0.5 * np.dot(w, w)
                pobj_avg[t] += 0.5 * np.dot(w_avg, w_avg)
            gradient = X[idx, :] * (np.dot(X[idx], w) - y[idx])
        elif loss == "hinge":
            hinge_loss = np.maximum(0., 1. - y * np.dot(X, w))
            hinge_loss_avg = np.maximum(0., 1. - y * np.dot(X, w_avg))
            pobj[t] = np.mean(hinge_loss)
            pobj_avg[t] = np.mean(hinge_loss_avg)
            gradient = X[idx] * (-y[idx] * (hinge_loss[idx] > 0.))
        if alpha > 0:
            gradient += alpha * w
        w -= gamma_func(t) * gradient
        # cf. http://research.microsoft.com/pubs/192769/tricks-2012.pdf
        mu = 1. / np.maximum(1, t - t0)
        w_avg = w_avg + mu * (w - w_avg)
        # w_avg = float(t) / (t + 1) * w_avg + 1. / (t + 1) * w
        all_w[t] = w
    # if average is True:
    #     w = np.mean(all_w, axis=0)
    return w, w_avg, all_w, pobj, pobj_avg


###############################################################################
#            Toy dataset
###############################################################################

n_samples = 1000
n_features = 100
n_iter = 10 * n_samples  # number of iterations
gamma = 0.1  # step size

def gamma(t):
    cst = 0.01
    return cst / (1 + cst * t)

X_toy = np.random.randn(n_samples, n_features)
epsilon_toy = np.random.randn(n_samples)
w_target = np.ones(n_features)
y_toy = X_toy.dot(w_target) + epsilon_toy


# Initialize w with just zeros
w_ini = np.zeros(X_toy.shape[1] + 1)

loss = 'mse'
# loss = 'hinge'  # QUESTION 4

w, w_avg, all_w, pobj, pobj_avg = stochastic_gradient(X_toy, y_toy, gamma,
                                                      n_iter, w_ini, loss=loss,
                                                      alpha=0)

plt.figure()
plt.plot(pobj, label="normal")
plt.plot(pobj_avg, 'k', label="avg", linewidth="4")
plt.xlabel('t')
plt.ylabel('cost')
plt.title('%s stochastic (toy)' % loss)
plt.show()

###############################################################################
#            IRIS dataset
###############################################################################

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Make the problem binary
X = X[y < 2]
y = y[y < 2]
y[y == 0] = -1
X = X[:, :2]


def plot_2d(X, y, w=None):
    """ Plot in 2D the dataset data, colors and symbols according to the
    class given by the vector y (if given); the separating hyperplan w can
    also be displayed if asked"""
    plt.figure()
    symlist = ['o', 's', '+', 'x', 'D', '*', 'p', 'v', '-', '^']
    collist = ['blue', 'red', 'purple', 'orange', 'salmon', 'black', 'grey',
               'fuchsia']

    labs = np.unique(y)
    idxbyclass = [y == labs[i] for i in range(len(labs))]

    for i in range(len(labs)):
        plt.plot(X[idxbyclass[i], 0], X[idxbyclass[i], 1], '+',
                 color=collist[i % len(collist)], ls='None',
                 marker=symlist[i % len(symlist)])
    plt.ylim([np.min(X[:, 1]), np.max(X[:, 1])])
    plt.xlim([np.min(X[:, 0]), np.max(X[:, 0])])
    mx = np.min(X[:, 0])
    maxx = np.max(X[:, 0])
    if w is not None:
        plt.plot([mx, maxx], [mx * -w[1] / w[2] - w[0] / w[2],
                              maxx * -w[1] / w[2] - w[0] / w[2]],
                 "g", alpha=1.)
    plt.show()

############################################################################
#            Displaying labeled data
############################################################################

# plot_2d(X[:, :2], y)


gamma = 0.01  # step size
n_iter = 10000  # number of iterations


# Initialize w with just zeros
w_ini = np.zeros(X.shape[1] + 1)

loss = 'mse'
# loss = 'hinge'  # QUESTION 4

w, w_avg, all_w, pobj, pobj_avg = stochastic_gradient(X, y, gamma, n_iter,
                                                      w_ini, loss=loss,
                                                      alpha=0)

plot_2d(X, y, w)
plt.title('%s stochastic' % loss)

plt.figure()
plt.plot(pobj, label="normal")
plt.plot(pobj_avg, 'k', label="avg", linewidth="4")
plt.xlabel('t')
plt.ylabel('cost')
plt.title('%s stochastic Iris' % loss)
plt.show()

############################################################################
#            Using Scikit-Learn
############################################################################

# QUESTION 6 : compare with Scikit-Learn

clf = linear_model.SGDClassifier(loss='hinge', penalty='none',
                                 n_iter=n_iter, shuffle=True,
                                 learning_rate='constant',
                                 eta0=gamma)
clf.fit(X, y)

plot_2d(X, y, np.c_[clf.intercept_, clf.coef_][0])
plt.title('Scikit-Learn')
