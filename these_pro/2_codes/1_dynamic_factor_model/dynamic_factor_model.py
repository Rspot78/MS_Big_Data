# note: this is a class to implement the dynamic factor model of
# Giannone, Reichlin and Small: "Nowcasting: the real-time information content
# of macroeconomic data", Journal of Monetary Economics 55 (2008)
# notations and algorithms are as in the paper and its appendix


# imports
import pandas as pd
import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt


# main class DynamicFactorModel
class DynamicFactorModel:

          
    def __init__(self, feature_dataframe, factors = 2, shocks = 2):
        self.features = feature_dataframe
        self.gdp = feature_dataframe['quarterly_gdp'].dropna()
        self.feature_names = feature_dataframe.columns[:-1].to_list()
        self.feature_dates = feature_dataframe.index
        self.gdp_dates = feature_dataframe['quarterly_gdp'].dropna().index
        self.r = factors
        self.q = shocks
        

    def train_raw_factors(self):
        # unpack
        features = self.features
        feature_dates = self.feature_dates
        r = self.r
        # get balanced feature panel: exclude gdp and any period with nan
        X = features.iloc[:, :-1].dropna().to_numpy()
        # normalize and create variance-covariance matrix
        X_mean, X_std = np.mean(X,0), np.std(X, 0)
        Z = (X - X_mean) / X_std
        T, n = Z.shape[0], Z.shape[1]
        S = Z.T @ Z / T
        # run pca to estimate the r dynamic factors
        eigenvalues, eigenvectors = nla.eig(S)
        D = np.diag(eigenvalues[:r])
        V = eigenvectors[:,:r]
        F =  Z @ V
        F_mean, F_std = np.mean(F,0), np.std(F, 0)
        # pass to dataframe
        columns = ['factor_' + str(i+1) for i in range(r)]
        raw_factors = pd.DataFrame(np.nan, index = feature_dates,
                               columns = columns)
        for i in range(r):
            raw_factors['factor_' + str(i+1)].iloc[:T] = F[:,i]
        # save as attributes
        self.X = X
        self.X_mean = X_mean
        self.X_std = X_std
        self.Z = Z
        self.T = T
        self.n = n
        self.S = S
        self.D = D
        self.V = V
        self.F = F
        self.F_mean = F_mean
        self.F_std = F_std 
        self.raw_factors = raw_factors

        
    def plot_raw_factors(self):
        # unpack
        r = self.r
        features = self.features
        gdp = self.gdp
        raw_factors = self.raw_factors
        gdp_dates = self.gdp_dates
        # create figure
        columns = r + 1
        rows = columns // 4 + 1
        fig = plt.figure(figsize = (18, 4 * rows))
        # plot gdp
        plt.suptitle('real gdp growth and structural factors', 
                     y = 1.02, fontsize = 18, fontweight = 'semibold')
        plt.subplot(rows, 4, 1)
        plt.plot(gdp_dates, gdp, linewidth = 1.5, color = (0.6, 0.1, 0.1))
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
        plt.xlim(gdp_dates[0], gdp_dates[-1])
        plt.grid(True)
        plt.title('real gdp growth')
        # plot factors
        factor_dates = raw_factors.index
        for i in range(r):
            factor_name = raw_factors.columns[i]            
            factor_data = raw_factors[factor_name]
            plt.subplot(rows, 4, i+2)
            plt.plot(factor_dates, factor_data, linewidth = 1.5, color = (0.6, 0.1, 0.1))
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(factor_dates[0], factor_dates[-1])
            plt.grid(True)
            plt.title(factor_name)        
        plt.show()        


    def train_parameters(self):
        # unpack
        V = self.V
        S = self.S
        D = self.D
        F = self.F
        T = self.T
        q = self.q
        # estimate the factor loadings
        Lambda_hat = V
        # estimate the covariance matrix of the idiosyncratic component
        psi_hat = np.diag(S - V @ D @ V.T)
        # estimate the VAR coefficients on the factors
        F_t = F[1:, :]
        F_t1 = F[:-1, :]
        sum_1 = F_t.T @ F_t1
        sum_2 = F_t1.T @ F_t1
        A_hat = nla.solve(sum_2.T, sum_1.T).T
        # estimate the variance-covariance matrix of factor shocks
        sum_3 = F_t.T @ F_t / (T - 1)
        sum_4 = A_hat @ (F_t1.T @ F_t1 / (T - 1)) @ A_hat.T
        Sigma_hat = sum_3 - sum_4
        eigenvalues, eigenvectors = nla.eig(Sigma_hat)
        P_hat = np.diag(eigenvalues[:q])
        M_hat = eigenvectors[:,:q]
        B_hat = M_hat @ np.sqrt(P_hat)
        # save as attributes        
        self.Lambda_hat = Lambda_hat
        self.psi_hat = psi_hat
        self.A_hat = A_hat
        self.B_hat = B_hat
        
        
    def train_smoothed_factors(self):
        # unpack
        features = self.features
        feature_dates = self.feature_dates
        feature_names = self.feature_names
        X_mean = self.X_mean
        X_std = self.X_std
        r = self.r
        T = self.T
        n = self.n
        A_hat = self.A_hat
        B_hat = self.B_hat
        Lambda_hat = self.Lambda_hat
        psi_hat = self.psi_hat
        F_mean = self.F_mean
        F_std = self.F_std
        # prepare observations for the Kalman smoother
        X = features.iloc[:,:-1].to_numpy()
        Z = (X - X_mean) / X_std
        Z_hat = np.zeros((T, n))
        F_hat = np.zeros((T, r))
        # initiate the Kalman filter
        A = A_hat
        Ups = B_hat @ B_hat.T
        Lbda = Lambda_hat
        psi = psi_hat
        F_tt = F_mean
        Ups_tt = np.diag(3 * F_std)
        # run the Kalman filter
        for period in range(T):
            # update z_t and Psi
            z_t = Z[period,:].copy()
            Psi = psi.copy()
            # replace any nan in z_t with 0, and set large variance
            if np.isnan(z_t).any():
                nan_positions = np.argwhere(np.isnan(z_t)).flatten()
                z_t[nan_positions] = 0
                Psi[nan_positions] = 10000
            Psi = np.diag(Psi)
            # run the Kalman steps
            F_t1t1 = F_tt
            Ups_t1t1 = Ups_tt
            F_tt1 = A @ F_t1t1
            Ups_tt1 = A @ Ups_t1t1 @ A.T + Ups
            z_tt1 = Lbda @ F_tt1
            Psi_tt1 = Lbda @ Ups_tt1 @ Lbda.T + Psi
            Phi_t = (nla.solve(Psi_tt1.T, Lbda) @ Ups_tt1.T).T
            F_tt = F_tt1 + Phi_t @ (z_t - z_tt1)
            Ups_tt = Ups_tt1 - Phi_t @ Psi_tt1 @ Phi_t.T
            F_hat[period,:] = F_tt
            Z_hat[period,:] = z_tt1
        # obtain feature predictions by de-normalizing
        X_hat = Z_hat * X_std + X_mean
        # pass to dataframe
        smoothed_features = pd.DataFrame(index = feature_dates)
        for i in range(n):
            smoothed_features[feature_names[i]] = X_hat[:,i]
        smoothed_factors = pd.DataFrame(index = feature_dates)
        for i in range(r):
            smoothed_factors['factor_' + str(i+1)] = F_hat[:,i]     
        # save as attributes
        self.F_hat = F_hat
        self.Z_hat = Z_hat
        self.X_hat = X_hat
        self.Ups_tt = Ups_tt
        self.smoothed_features = smoothed_features 
        self.smoothed_factors = smoothed_factors        


    def plot_smoothed_factors(self):
        # unpack
        r = self.r
        feature_dates = self.feature_dates
        raw_factors = self.raw_factors
        smoothed_factors = self.smoothed_factors     
        # plot factors
        columns = r
        rows = columns // 2 + 1
        fig = plt.figure(figsize = (14, 6 * rows))
        plt.suptitle('structural factors: raw and smoothed estimates', 
                     y = 0.95, fontsize = 18, fontweight = 'semibold')
        for i in range(r):            
            data_raw = raw_factors['factor_' + str(i+1)]
            data_smoothed = smoothed_factors['factor_' + str(i+1)]            
            plt.subplot(rows, 2, i+1)
            plt.plot(feature_dates, data_raw, linewidth = 2, color = (0.1, 0.3, 0.7),
                     label = 'raw factors')
            plt.plot(feature_dates, data_smoothed, linewidth = 2, 
                     color = (0.3, 0.7, 0.1), linestyle = 'dashed', 
                     label = 'smoothed estimates')            
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(feature_dates[0], feature_dates[-1])
            plt.legend()
            plt.grid(True)
            plt.title('factor_' + str(i+1))        
        plt.show()          

    
    def plot_smoothed_features(self):
        # unpack
        n = self.n
        features = self.features.iloc[:,:-1]
        feature_dates = self.feature_dates
        feature_names = self.feature_names
        smoothed_features = self.smoothed_features
        # plot predicted features
        columns = n
        rows = columns // 3 + 1
        fig = plt.figure(figsize = (18, 6 * rows))
        plt.suptitle('features: raw and predicted', 
                     y = 0.90, fontsize = 18, fontweight = 'semibold')
        for i in range(n):           
            data_actual = features[feature_names[i]]
            data_predicted = smoothed_features[feature_names[i]]         
            plt.subplot(rows, 3, i+1)
            plt.plot(feature_dates, data_actual, linewidth = 2, color = (0.1, 0.3, 0.7),
                     label = 'actual')
            plt.plot(feature_dates, data_predicted, linewidth = 2, 
                     color = (0.3, 0.7, 0.1), linestyle = 'dashed', 
                     label = 'predicted')            
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(feature_dates[0], feature_dates[-1])
            plt.legend()
            plt.grid(True)
            plt.title(feature_names[i])        
        plt.show()
        
        
    def train_gdp_regression(self):
        # unpack
        smoothed_factors = self.smoothed_factors
        features = self.features
        feature_dates = self.feature_dates
        gdp_dates = self.gdp_dates
        # recover smoothed factors, concatenate gdp, trim months with NaNs
        regression_dataframe = smoothed_factors.copy()
        regression_dataframe['quarterly_gdp'] = features['quarterly_gdp']
        regression_dataframe = regression_dataframe.dropna()
        regression_dates = regression_dataframe.index
        # extract regressors and estimate model
        F = np.array(regression_dataframe.iloc[:,:-1])
        F = np.concatenate((np.ones((len(F),1)), F), axis=1)
        y = np.array(regression_dataframe['quarterly_gdp'])
        beta = nla.solve(F.T @ F, F.T @ y)
        # get predictions over sample periods, then keep only quarters
        F = np.array(smoothed_factors.copy())
        F = np.concatenate((np.ones((len(F),1)), F), axis=1)
        y_hat = F @ beta
        y_hat = y_hat[feature_dates.month % 3 == 0]
        # pass to dataframe
        for index, date in enumerate(gdp_dates):
            self.smoothed_features.loc[date, 'quarterly_gdp'] = y_hat[index]  
        # save as attributes            
        self.beta = beta
        self.y_hat = y_hat
        
        
    def plot_smoothed_gdp(self):
        # unpack
        gdp_dates = self.gdp_dates
        gdp = self.gdp
        smoothed_features = self.smoothed_features
        # plot gdp predictions
        fig = plt.figure(figsize = (10, 8))
        plt.title('gdp: actual and predicted', 
                     y = 1.02, fontsize = 18, fontweight = 'semibold')
        gdp_predicted = smoothed_features.loc[gdp_dates, 'quarterly_gdp']       
        plt.plot(gdp_dates, gdp, linewidth = 2, 
                 color = (0.7, 0.1, 0.3), label = 'actual')
        plt.plot(gdp_dates, gdp_predicted, linewidth = 2, 
                 color = (0.2, 0.6, 0.1), linestyle = 'dashed', 
                 label = 'predicted')            
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
        plt.xlim(gdp_dates[0], gdp_dates[-1])
        plt.legend()
        plt.grid(True)      
        plt.show()
        
        
    def nowcast_gdp(self, h):
        # unpack
        features = self.features
        n = self.n
        r = self.r
        A_hat = self.A_hat
        B_hat = self.B_hat
        Lambda_hat = self.Lambda_hat
        psi_hat = self.psi_hat
        F_hat = self.F_hat
        Ups_tt = self.Ups_tt
        X_std = self.X_std
        X_mean = self.X_mean
        beta = self.beta
        feature_names = self.feature_names
        smoothed_features = self.smoothed_features
        smoothed_factors = self.smoothed_factors
        # extract last month of the sample, and calculate additional months
        year = features.index[-1].year
        month = features.index[-1].month
        if month == 3 or month == 6 or month == 9 or month == 12:
            months = 3
        elif month == 2 or month == 5 or month == 8 or month == 11:
            months = 1
        elif month == 1 or month == 4 or month == 7 or month == 10:
            months = 2
        if month == 12:
            month = 0
            year += 1
        months += 3 * (h - 1)
        # initiate the Kalman filter to get predictions
        Z_predicted = np.zeros((months, n))
        F_predicted = np.zeros((months, r))        
        A = A_hat
        Ups = B_hat @ B_hat.T
        Lbda = Lambda_hat
        psi = psi_hat
        F_tt = F_hat[-1,:]
        Ups_tt = Ups_tt
        Psi = np.diag(psi.copy())
        # run the Kalman filter
        for period in range(months):
            F_t1t1 = F_tt
            Ups_t1t1 = Ups_tt
            F_tt1 = A @ F_t1t1
            Ups_tt1 = A @ Ups_t1t1 @ A.T + Ups
            z_tt1 = Lbda @ F_tt1
            Psi_tt1 = Lbda @ Ups_tt1 @ Lbda.T + Psi
            F_tt = F_tt1
            Ups_tt = Ups_tt1
            F_predicted[period,:] = F_tt
            Z_predicted[period,:] = z_tt1
        X_predicted = Z_predicted * X_std + X_mean
        F = np.concatenate((np.ones((months,1)), F_predicted), axis=1)
        y_predicted = F @ beta
        # pass to dataframe      
        dates = pd.date_range(str(year) + '-' + str(month + 1),
            periods = months, freq='M')
        columns = feature_names
        predicted_features = pd.DataFrame(X_predicted, index = dates, 
                                      columns = columns)
        predicted_features['quarterly_gdp'] = y_predicted
        columns = ['factor_' + str(i+1) for i in range(r)]
        predicted_factors = pd.DataFrame(F_predicted, index = dates, 
                                         columns = columns)
        smoothed_and_predicted_features = pd.concat([smoothed_features, 
                                                 predicted_features])
        smoothed_and_predicted_factors = pd.concat([smoothed_factors, 
                                         predicted_factors])
        # calculate specifically the nowcasts for the quarters of interest
        nowcast_quarters = [t+2 for t in range(-3 * h, -1, 3)]
        nowcasts_gdp = smoothed_and_predicted_features \
            .iloc[nowcast_quarters,:]['quarterly_gdp']
        # save as attributes 
        self.h = h        
        self.F_predicted = F_predicted
        self.Z_predicted = Z_predicted
        self.X_predicted = X_predicted 
        self.y_predicted = y_predicted 
        self.predicted_features = predicted_features
        self.predicted_factors = predicted_factors        
        self.smoothed_and_predicted_features = smoothed_and_predicted_features
        self.smoothed_and_predicted_factors = smoothed_and_predicted_factors
        self.nowcasts_gdp = nowcasts_gdp
        
        
    def plot_nowcast_features(self):
        # unpack
        features = self.features
        feature_dates = self.feature_dates
        smoothed_and_predicted_features = self.smoothed_and_predicted_features
        feature_names = self.feature_names
        n = self.n
        h = self.h
        # plot feature nowcasts
        dates_actual = feature_dates
        dates_nowcast = smoothed_and_predicted_features.index
        columns = n
        rows = columns // 3 + 1
        fig = plt.figure(figsize = (18, 6 * rows))
        plt.suptitle('features: raw and nowcast', 
                     y = 0.90, fontsize = 18, fontweight = 'semibold')
        for index, feature in enumerate(feature_names):           
            feature_actual = features[feature]
            feature_nowcast = smoothed_and_predicted_features[feature]         
            plt.subplot(rows, 3, index+1)
            plt.plot(dates_actual, feature_actual, linewidth = 2,
                     color = (0.1, 0.3, 0.7), label = 'actual')
            plt.plot(dates_nowcast, feature_nowcast, linewidth = 2, 
                     color = (0.3, 0.7, 0.1), linestyle = 'dashed', 
                     label = 'predicted')            
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(dates_nowcast[- 3 * h - 12], dates_nowcast[-1])
            plt.legend()
            plt.grid(True)
            plt.title(feature)        
        plt.show()
        
          
    def plot_nowcast_gdp(self):
        # unpack
        gdp = self.gdp
        gdp_dates = self.gdp_dates
        smoothed_and_predicted_features = self.smoothed_and_predicted_features
        h = self.h
        # plot gdp nowcasts
        gdp_actual = gdp
        dates_actual = gdp_dates
        gdp_nowcast = smoothed_and_predicted_features['quarterly_gdp'] \
                                                        .dropna()
        dates_nowcast = gdp_nowcast.index
        nowcast_quarters = [t+2 for t in range(-3 * h, -1, 3)]
        fig = plt.figure(figsize = (10, 8))
        plt.title('gdp: raw and nowcast', 
                     y = 1.02, fontsize = 18, fontweight = 'semibold')
        plt.plot(dates_actual, gdp_actual, linewidth = 2, 
                 color = (0.7, 0.1, 0.3), label = 'actual')
        plt.plot(dates_nowcast, gdp_nowcast, linewidth = 2, 
                 color = (0.2, 0.6, 0.1), linestyle = 'dashed', \
                 label = 'predicted', marker = 'D', markersize = 10 , \
                 markevery = nowcast_quarters)            
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
        plt.xlim(dates_nowcast[- 3 * h - 12], dates_nowcast[-1])
        plt.legend()
        plt.grid(True)      
        plt.show()
        for date in dates_nowcast[nowcast_quarters]:
            quarter = date.month / 3
            year = date.year
            value = round(gdp_nowcast.loc[date], 3)
            print('The nowcast for ' + str(year) + 'Q' + str(quarter) + ' is ' 
                  + str(value) + '%.')
            
  
    def load(self, path, feature_file, information_file, gdp_file):
        self.load_data(path, feature_file, information_file, gdp_file)
        self.process_data()
        
        
    def train(self):
        self.estimate_factors()
        self.estimate_parameters()
        self.estimate_smoothed_factors()
        self.estimate_gdp_prediction_model()
        
    
    def predict(self, h):
        self.nowcast_gdp(h)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        