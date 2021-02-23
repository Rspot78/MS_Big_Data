# note: this is a class to implement the midas regression model of
# Ghysels et al. (2004): "The MIDAS touch: Mixed data sampling 
# regression models", CIRANO Working Papers 2004s-20, CIRANO
# notations follow those used in the internship report


import pandas as pd
import numpy as np
import numpy.linalg as nla
import scipy.optimize as sco


# main class MidasRegression
class MidasRegression:
    
    
    def __init__(self, feature_dataframe, h = 1, p = 2, q = 6, m = 3, factorization = 'amon'):
        self.features = feature_dataframe
        self.h = h
        self.p = p
        self.q = q
        self.m = m
        self.factorization = factorization
        
        
    def make_regressors(self):
        # unpack values
        features = self.features
        h, p, q, m = self.h, self.p, self.q, self.m
        # associate data
        raw_features = features.to_numpy()
        # indices of rows with quarterly entries
        quarterly_indices = np.where(~np.isnan(raw_features[:,-1]))[0]
        # index of first row with enough quaterly lags to account for h and p
        quarterly_row = quarterly_indices[p + h - 1]
        # index of first row with enough monthly lags to account for h and q
        monthly_row = quarterly_indices[np.searchsorted(quarterly_indices, q + m * h - 1)]
        # first row to account for quarterly lags, monthly lags and steps ahead
        first_row = max(quarterly_row, monthly_row)
        # last row to obtain a quarterly value
        last_row = quarterly_indices[-1]
        # generate y, Y and X
        y = self.generate_y(raw_features, first_row, last_row)
        Y = self.generate_Y(raw_features, first_row, last_row)
        X_i = self.generate_X_i(raw_features, first_row, last_row)
        T, n = y.shape[0], raw_features.shape[1] - 1
        self.data_matrix = raw_features[first_row:last_row+1,:-1]
        self.y = y
        self.Y = Y
        self.X_i = X_i
        self.T = T
        self.n = n
        
        
    def generate_y(self, raw_data, first_row, last_row):
        y = raw_data[first_row:last_row+1,-1]
        y = y[~np.isnan(y)]
        return y
    

    def generate_Y(self, raw_data, first_row, last_row):
        h, p, m = self.h, self.p, self.m
        y = raw_data[:,-1]
        Y = np.zeros(((last_row - first_row) // m + 1, p))
        for i in range(p):
            temp = y[first_row - m * (h + i): last_row - m * (h + i) + 1]
            temp = temp[~np.isnan(temp)]
            Y[:,i] = temp
        return Y
        
    
    def generate_X_i(self, raw_data, first_row, last_row):
        h, q, m = self.h, self.q, self.m
        n = raw_data.shape[1] - 1
        X_i = np.zeros(((last_row - first_row) // m + 1, q, 0))
        for i in range(n):    
            x = raw_data[:,i]
            x_i = np.zeros(((last_row - first_row) // m + 1, q))
            for i in range(q):
                temp = x[first_row - (m * h) - i: last_row - (m * h) - i + 1]
                temp = temp[::m]
                x_i[:,i] = temp
            x_i = x_i[:, :, np.newaxis]
            X_i = np.dstack((X_i, x_i))
        return X_i        
        
        
    def train(self):
        # unpack
        factorization, y, n, p = self.factorization, self.y, self.n, self.p
        # initiate parameter values for theta
        if factorization == 'amon':
            theta_0 = -0.05 * np.ones(n)
            bound = [(-1, 0)] * n
        elif factorization == 'beta':
            theta_0 = 5 * np.ones(n)
            bound = [(1, None)] * n
        # find theta that maximizes the log likelihood of the model
        result = sco.minimize(self.log_likelihood, theta_0, \
                                          method='L-BFGS-B',  bounds = bound)
        theta = result.x
        success = result.success
        if success:
            print('Optimization successful: theta properly estimated.')
        else:
            print('Optimization failed: theta may be ill-specified.')
        # conditional on theta, build X
        X = self.generate_X(theta)
        # estimate delta by OLS
        delta = nla.solve(X.T @ X, X.T @ y)
        mu = delta[0]
        gamma = delta[1:1+p]
        beta = delta[1+p:1+p+n]
        # save as attributes
        self.mu = mu
        self.gamma = gamma
        self.beta = beta
        self.theta = theta        
        
        
    def log_likelihood(self, theta):
        # unpack
        y = self.y
        # build X
        X = self.generate_X(theta)
        # compute the log-likelihood (the negative in fact, to use a minimizer)
        Xy = X.T @ y
        log_likelihood = - Xy @ nla.solve(X.T @ X, Xy)
        return log_likelihood

            
    def generate_X(self, theta):
        # unpack
        y, Y, X_i, n, T = self.y, self.Y, self.X_i, self.n, self.T
        # start building X
        X = np.hstack((np.ones((T, 1)), Y))
        # recover the weights from theta and continue building X
        for i in range(n):
            # recover the weights from theta
            w_i = self.weights(theta[i])
            # build X_i_tilde
            X_i_tilde = X_i[:,:,i] @ w_i
            X_i_tilde = np.reshape(X_i_tilde, (-1,1))
            X = np.hstack((X, X_i_tilde))
        return X        
        
        
    def weights(self, theta):
        # unpack
        q, factorization = self.q, self.factorization
        # generate weights in the case of Amon factorization
        J = np.arange(1, q+1)
        if factorization == 'amon':
            raw_w = np.exp(theta * J * J)
        elif factorization == 'beta':
            raw_w = (1 - J / q) ** (theta - 1)
        w = raw_w / np.sum(raw_w)
        return w        
        

    def predict(self):
        # unpack
        n, p, q = self.n, self.p, self.q
        y, data_matrix = self.y, self.data_matrix
        mu, gamma, beta, theta = self.mu, self.gamma, self.beta, self.theta
        # generate prediction sequentially using regression formula
        y_hat = 0
        # constant
        y_hat += mu
        # low frequency lags
        Y = np.flip(y[-p:])
        y_hat += Y @ gamma
        # high frequency lags
        for i in range(n):
            beta_i = beta[i]
            x_i = np.flip(data_matrix[-q:,i])
            theta_i = theta[i]            
            w_i = self.weights(theta_i)
            y_hat += beta_i * (x_i @ w_i)
        return y_hat        
        
        
        
        
        
        
        
        
        
        
        
        
        