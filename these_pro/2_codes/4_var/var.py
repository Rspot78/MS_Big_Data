# imports
import numpy as np
import numpy.linalg as nla
import pandas as pd
import matplotlib.pyplot as plt


# main class Vector_Autoregression
class Vector_Autoregression:


    def __init__(self, feature_dataframe):
        features = feature_dataframe
        dates = features.index
        feature_list = features.columns.to_list()
        # get variables and dimensions        
        n = len(feature_list)
        T = len(features)
        # get data as array
        y = features.to_numpy()  
        # save as attribute
        self.features = features        
        self.feature_list = feature_list
        self.dates = dates
        self.n = n
        self.T = T
        self.y = y


    def regressors(self, p):
        # unpack
        y = self.y
        # create regressors
        Y, X = self.create_lag_matrix(y, p)
        # save as attribute
        self.Y = Y 
        self.X = X
        self.p = p
        
        
    def create_lag_matrix(self, A, p):
        n, T = A.shape[1], A.shape[0]
        Y = A[p:, :]
        X = np.ones((T - p, 1))
        for lag in range(p):
            X = np.concatenate((X, A[p-lag-1:T-lag-1, :]), axis = 1)
        return Y, X


    def train(self):
        Y, X = self.Y, self.X
        B = nla.solve(X.T @ X, X.T @ Y)
        # save as attribute
        self.B = B


    def forecast(self, h):
        # unpack
        Y, B, n, p = self.Y, self.B, self.n, self.p
        # initiate predictions
        for t in range(h):
            # recover regressors
            Y_predict = np.vstack((Y[-p:,:], np.zeros((1,n))))
            y, x = self.create_lag_matrix(Y_predict, p)
            # create prediction
            y_hat = x @ B
            Y = np.vstack((Y, y_hat))
        Y_hat = Y[-h:,:]
        # save as attributes
        self.h = h
        self.Y_hat = Y_hat


    def plot_forecast(self):
        # unpack elements for plots
        y, Y_hat, n, T, h = self.y, self.Y_hat, self.n, self.T, self.h
        feature_list, dates = self.feature_list, self.dates
        plot_dates = pd.date_range(start = dates[0], periods = T + h, freq ='Q')
        # get plot data for each variable
        plot_values = np.full((T + h, 2, n), np.nan)
        for i in range(n):
            plot_values[:T,0,i] = y[:,i]
            plot_values[T-1,1,i] = y[-1,i]
            plot_values[T:,1,i] = Y_hat[:,i]
        # plot, looping over variables:
        fig = plt.figure(figsize=(16,18))
        plt.suptitle('Feature predictions', y=0.93, fontsize=18, \
                     fontweight='semibold')
        rows, columns = n // 3 + 1, 3    
        for i in range(n):
            # remove all-nan rows and plot
            ax = plt.subplot(rows, columns, i+1)            
            plt.plot(plot_dates, plot_values[:,0,i], linewidth = 2, color = (0.1, 0.3, 0.7))           
            plt.plot(plot_dates, plot_values[:,1,i], linewidth = 2, color = (0.2, 0.6, 0.2))
            plt.title(feature_list[i]) 
            plt.xlim(plot_dates[0], plot_dates[-1])
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.grid(True)
        plt.show()




























