# imports
import numpy as np
import pandas as pd
import sklearn.preprocessing as spr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt



# main class Lstm_Var
class Lstm_Var:
    
    
    def __init__(self, feature_dataframe):
        features = feature_dataframe
        dates = features.index
        feature_list = features.columns.to_list() 
        # save as attribute
        self.features = features
        self.dates = dates        
        self.feature_list = feature_list

        
    def regressors(self, p):
        # unpack
        features = self.features
        # get data as array
        y = features.to_numpy() 
        Y, X = self.lag_matrix(y, p)
        n, T, k = Y.shape[1], Y.shape[0], X.shape[1]
        # standardize data as lstm models are sensitive to scale
        X_scaler = spr.StandardScaler()
        X = X_scaler.fit_transform(X)
        Y_scaler = spr.StandardScaler()
        Y = Y_scaler.fit_transform(Y)
        # lstm expects data to be of format [samples, timesteps, features]
        X = X.reshape((T, 1, k))
        # save as attributes
        self.Y = Y
        self.X = X
        self.p = p
        self.n = n
        self.T = T
        self.k = k
        self.X_scaler = X_scaler
        self.Y_scaler = Y_scaler


    def lag_matrix(self, A, p):
        n, T = A.shape[1], A.shape[0]
        Y = A[p:, :]
        X = np.zeros((T - p, 0))
        for lag in range(p):
            X = np.concatenate((X, A[p-lag-1:T-lag-1, :]), axis = 1)
        return Y, X
    

    def grid_search(self, layer_grid, unit_grid):
        # unpack
        Y, X, k, n = self.Y, self.X, self.k, self.n
        # split the data in train (75%) and test (25%)
        Y_train = np.vstack([Y[i::4, :] for i in range(3)])
        X_train = np.vstack([X[i::4, :, :] for i in range(3)])
        Y_test = Y[3::4,:]
        X_test = X[3::4,:,:]
        # run the grid search
        min_loss = 10000
        for layer in layer_grid:
            for unit in unit_grid:
                model = Sequential()
                if layer == 1:
                    model.add(LSTM(units = unit, input_shape = (1, k)))
                else:
                    model.add(LSTM(units = unit, input_shape = (1, k), return_sequences = True))
                    model.add(Dropout(0.25))
                    for i in range(1,layer-1):
                        model.add(LSTM(units = unit, return_sequences = True))
                        model.add(Dropout(0.25))
                    model.add(LSTM(units = unit))
                    model.add(Dropout(0.25))                    
                model.add(Dense(units = n, activation = 'linear'))
                model.compile(loss = 'mean_squared_error', optimizer = 'adam')
                history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 10, verbose = False)
                loss = history.history['val_loss'][-1]
                if loss < min_loss:
                    min_loss = loss
                    optimal_layer = layer
                    optimal_unit = unit
        print('optimal number of layers: ' + str(optimal_layer))
        print('optimal number of units: ' + str(optimal_unit))
        return optimal_layer, optimal_unit


    def train(self, layers, units, epochs):
        # unpack
        Y, X, n, k = self.Y, self.X, self.n, self.k
        # set the model
        model = Sequential()
        for i in range(layers):
            model.add(LSTM(units = units, input_shape = (1, k)))
            model.add(Dropout(0.25))
        model.add(Dense(units = n, activation = 'linear'))
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        model.summary()
        # train the model
        model.fit(X, Y, epochs = epochs)
        # save as attribute
        self.model = model                
         
        
    def forecast(self, h):
        # unpack
        Y, n, p = self.Y, self.n, self.p
        model = self.model
        Y_scaler = self.Y_scaler
        # loop over prediction periods and predict recursively
        for t in range(h):
            # recover regressors
            Y_p = np.vstack((Y[-p:,:], np.zeros((1,n))))
            _, x = self.lag_matrix(Y_p, p)
            x = x.reshape(1, 1, -1)
            y = model.predict(x, verbose = 0)
            # pass predictions as regressors for next period
            Y = np.vstack((Y, y))
        Y_hat = Y[-h:,:]
        # unscale
        Y_scaler.inverse_transform(Y_hat)
        # store as attributes
        self.h = h
        self.Y_hat = Y_hat
        
        
    def plot_forecast(self):
        # unpack elements for plots
        Y, n, T, h = self.Y, self.n, self.T, self.h
        feature_list, dates = self.feature_list, self.dates
        Y_hat = self.Y_hat
        plot_dates = pd.date_range(start = dates[0], periods = T + h, freq ='Q')
        # get plot data for each variable
        plot_values = np.full((T + h, 4, n), np.nan)
        for i in range(n):
            plot_values[:T,0,i] = Y[:,i]
            plot_values[T-1,1,i] = Y[-1,i]
            plot_values[T:,1,i] = Y_hat[:,i]
        # plot, looping over variables:
        fig = plt.figure(figsize=(16,18))
        plt.suptitle('Feature predictions', \
                     y=0.93, fontsize=18, fontweight='semibold')
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















