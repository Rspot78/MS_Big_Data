# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RFR


# main class Random_Forest_Midas
class Random_Forest_Midas:
    
    
    def __init__(self, feature_dataframe, h=1, p=2, q=6, m=3):
        features = feature_dataframe
        dates = features.index
        feature_list = features.columns.to_list()
        n = len(feature_list) - 1
        # save as attribute
        self.features = features
        self.h = h
        self.p = p
        self.q = q
        self.m = m
        self.n = n
        self.dates = dates        
        self.feature_list = feature_list


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
        # generate y and X
        y = self.generate_y(raw_features, first_row, last_row)
        Y = self.generate_Y(raw_features, first_row, last_row)
        X_i = self.generate_X_i(raw_features, first_row, last_row)
        X = self.generate_X(Y, X_i)
        T, n = y.shape[0], X_i.shape[2]
        self.feature_array = raw_features[first_row:last_row+1,:-1]
        self.y = y
        self.X = X
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
    
    
    def generate_X(self, Y, X_i):
        # start building X
        X = Y.copy()
        # build sequentially with X_i
        for i in range(X_i.shape[2]):
            X = np.hstack((X, X_i[:,:,i]))
        return X
    
    
    def train(self, estimators, depth, split):
        # unpack
        y, X, n, p = self.y, self.X, self.n, self.p
        # estimate the random forest midas
        rf = RFR(n_estimators = estimators, max_depth = depth, min_samples_split = split)
        rf.fit(X, y)
        # save as attributes
        self.rf = rf  
    
    
    def predict(self):
        # unpack
        n, p, q = self.n, self.p, self.q
        rf = self.rf
        y, features = self.y, self.feature_array
        # low frequency lags        
        x = np.flip(y[-p:])
        # high frequency lags        
        for i in range(n):
            x = np.hstack((x, np.flip(features[-q:,i])))
        x = x.reshape(-1, 1).T
        y_hat = rf.predict(x) 
        self.y_hat = y_hat
    
    
 
    


# main class Random_Forest_Var
class Random_Forest_Var:
    
    
    def __init__(self, feature_dataframe):
        features = feature_dataframe.iloc[:,:-1]
        dates = features.index
        feature_list = features.columns.to_list()
        # get data as array
        y = features.to_numpy()
        # save as attribute
        self.features = features
        self.feature_list = feature_list
        self.dates = dates
        self.y = y
        
        
    def regressors(self, p):
        # unpack
        y = self.y
        # create regressors
        Y, X = self.lag_matrix(y, p)
        n, T = Y.shape[1], Y.shape[0]
        # save as attribute
        self.Y = Y 
        self.X = X
        self.p = p
        self.n = n
        self.T = T


    def lag_matrix(self, A, p):
        n, T = A.shape[1], A.shape[0]
        Y = A[p:, :]
        X = np.zeros((T - p, 0))
        for lag in range(p):
            X = np.concatenate((X, A[p-lag-1:T-lag-1, :]), axis = 1)
        return Y, X


    def train(self, estimators, depth, split):
        Y, X, n = self.Y, self.X, self.n
        regression_list = []
        for i in range(n):
            regression_list.append(RFR(n_estimators = estimators, max_depth = depth, min_samples_split = split))
            regression_list[i] = regression_list[i].fit(X, Y[:,i])
        # save as attribute
        self.regression_list = regression_list
        
        
    def forecast(self, h):
        # unpack
        Y, p, n = self.Y, self.p, self.n
        regression_list = self.regression_list
        # loop over prediction periods
        for t in range(h):
            y = np.zeros(n)
            x = np.empty(0)        
            for i in range(p):
                x = np.hstack((x, Y[-i-1,:]))          
            x = x.reshape(1, -1)
            for i in range(n):
                y[i] = regression_list[i].predict(x)
            Y = np.vstack((Y, y))
        Y_hat = Y[-h:,:]
        self.h = h
        self.Y_hat = Y_hat
        
        
    def plot_forecast(self):
        # unpack elements for plots
        Y, n, T, h = self.Y, self.n, self.T, self.h
        feature_list, dates = self.feature_list, self.dates
        Y_hat = self.Y_hat
        plot_dates = pd.date_range(start = dates[0], periods = T + h, freq ='M')
        # get plot data for each variable
        plot_values = np.full((T + h, 4, n), np.nan)
        for i in range(n):
            plot_values[:T,0,i] = Y[:,i]
            plot_values[T-1,1,i] = Y[-1,i]
            plot_values[T:,1,i] = Y_hat[:,i]
        # plot, looping over variables:
        fig = plt.figure(figsize=(16, 1.7 * n))
        plt.suptitle('Feature predictions', \
                     y=0.9, fontsize=18, fontweight='semibold')
        rows, columns = n // 3 + 1, 3     
        for i in range(n):
            # remove all-nan rows and plot
            ax = plt.subplot(rows, columns, i+1)           
            plt.plot(plot_dates, plot_values[:,0,i], linewidth = 2, color = (0.1, 0.3, 0.7))
            plt.plot(plot_dates, plot_values[:,1,i], linewidth = 2, color = (0.2, 0.6, 0.2))
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))            
            plt.title(feature_list[i]) 
            plt.xlim(plot_dates[0], plot_dates[-1])
            plt.grid(True)
        plt.show()      





