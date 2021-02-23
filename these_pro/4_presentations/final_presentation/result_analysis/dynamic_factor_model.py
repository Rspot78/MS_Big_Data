# note: this is a class to implement the dynamic factor model of
# Giannone, Reichlin and Small: "Nowcasting: the real-time information content
# of macroeconomic data", Journal of Monetary Economics 55 (2008)
# notations and algorithms are as in the paper and its appendix


# imports
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt


# main class DynamicFactorModel
class DynamicFactorModel:

          
    def __init__(self, factors = 2, shocks = 2):
        self.r = factors
        self.q = shocks


    def load_data(self, path, feature_file, information_file, gdp_file):
        # load monthly features and create dataframe with formated dates
        raw_features = pd.read_csv(path + '/' + feature_file, delimiter=',')
        raw_features = raw_features.set_index(
                pd.to_datetime(raw_features['date']
                .apply(lambda x: x.split('m')[0] + x.split('m')[1]), 
                format = "%Y%m") + MonthEnd(1)).drop('date', 1)
        # load information about monthly data and create dataframe
        information = pd.read_csv(path + '/' + information_file, 
                           delimiter=',', index_col=0)
        # load quarterly gdp data and create dataframe with formated dates
        gdp = pd.read_csv(path + '/' + gdp_file, delimiter=',')
        gdp = gdp.set_index(pd.to_datetime(gdp['date'] 
            .apply(lambda x: x.split('q')[0] + str(3 * int(x.split('q')[1]))), 
            format = "%Y%m") + MonthEnd(1)).drop('date', 1) \
            .rename(columns={"gdp": "quarterly_gdp"})
        gdp_dates = gdp.index.strftime("%Y-%m-%d")
        # concatenate raw features and quarterly gdp in a single dataframe
        raw_data = raw_features.copy()
        for date in gdp_dates:
            raw_data.loc[date, 'quarterly_gdp'] = gdp.loc[date, 
                                                          'quarterly_gdp']
        # save as attributes
        self.raw_features = raw_features        
        self.information = information        
        self.raw_data = raw_data
        
        
    def process_data(self):
        # initiate empty dataframe of processed features
        processed_features = pd.DataFrame(index =
                                          self.raw_features.index.copy())
        # loop over features and implement adequate transformation
        feature_list = self.raw_features.columns.to_list()
        for feature in feature_list:
            data = self.raw_features[feature]
            transformation = self.information['transformation'] \
            .loc[self.information['feature'] == feature].tolist()[0]
            if transformation == 1:
                processed_data = np.array(data)
            elif transformation == 2:
                processed_data = np.array(100 * np.log(data))          
            elif transformation == 3:
                processed_data = 100 * np.log(data)
                processed_data = np.array([processed_data[i]
                - processed_data[i-12] for i in range(len(processed_data))])
            processed_data = np.array([processed_data[i] + processed_data[i-1] 
                             + processed_data[i-2] - processed_data[i-3] 
                             - processed_data[i-4] - processed_data[i-5]
                               for i in range(len(processed_data))])
            processed_features[feature] = processed_data
        # trim the first 17 periods used for differencing computations        
        processed_features = processed_features.iloc[17:]
        # pass to dataframe and concatenate quarterly gdp (no transformation)
        processed_data = processed_features.copy()
        processed_data['quarterly_gdp'] = self.raw_data['quarterly_gdp'] \
                                            .iloc[17:]
        date_list = processed_features.index
        gdp_date_list = date_list[date_list.month % 3 == 0]
        nb_features, nb_periods = len(feature_list), len(date_list)
        # save as attributes
        self.processed_features = processed_features
        self.processed_data = processed_data
        self.feature_list = feature_list
        self.date_list = date_list
        self.gdp_date_list = gdp_date_list
        self.nb_features, self.nb_periods = nb_features, nb_periods
        
        
    def plot_processed_data(self):
        rows = self.nb_features // 4 + 1
        dates = self.date_list
        fig = plt.figure(figsize=(18, 4 * rows))
        plt.suptitle('processed features, as Q-to-Q growth rate', 
                     y = 0.9, fontsize = 18, fontweight = 'semibold')
        for i in range(self.nb_features):
            data = self.processed_features[self.feature_list[i]]
            plt.subplot(rows, 4, i+1)
            plt.plot(dates, data, linewidth = 1.5)
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(dates[0], dates[-1])
            plt.grid(True)
            plt.title(self.feature_list[i])        
        plt.show()         
        

    def estimate_factors(self):
        # get balanced feature panel: exclude gdp and periods with nan
        X = self.processed_data.iloc[:, :-1].dropna().to_numpy()
        # normalize and run pca to estimate the r dynamic factors
        X_mean, X_std = np.mean(X,0), np.std(X, 0)
        Z = (X - X_mean) / X_std
        T = Z.shape[0]
        S = Z.T @ Z / T
        eigenvalues, eigenvectors = nla.eig(S)
        D = np.diag(eigenvalues[:self.r])
        V = eigenvectors[:,:self.r]
        F =  Z @ V
        F_mean, F_std = np.mean(F,0), np.std(F, 0)
        # pass to dataframe
        columns = ['factor_' + str(i+1) for i in range(self.r)]
        raw_factors = pd.DataFrame(np.nan, index = self.date_list,
                               columns = columns)
        for i in range(self.r):
            raw_factors['factor_' + str(i+1)].iloc[:T] = F[:,i]
        # save as attributes
        self.X = X
        self.X_mean = X_mean
        self.X_std = X_std
        self.Z = Z
        self.T = T
        self.S = S
        self.D = D
        self.V = V
        self.F = F
        self.F_mean = F_mean
        self.F_std = F_std 
        self.raw_factors = raw_factors

        
    def plot_factors(self):
        columns = self.r + 1
        rows = columns // 4 + 1
        fig = plt.figure(figsize = (18, 4 * rows))
        plt.suptitle('real gdp growth and structural factors', 
                     y = 1.02, fontsize = 18, fontweight = 'semibold')
        data = self.processed_data['quarterly_gdp'].dropna()
        dates = data.index
        plt.subplot(rows, 4, 1)
        plt.plot(dates, data, linewidth = 1.5, color = (0.6, 0.1, 0.1))
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
        plt.xlim(dates[0], dates[-1])
        plt.grid(True)
        plt.title('real gdp growth')
        dates = self.raw_factors.index
        for i in range(self.r):
            name = self.raw_factors.columns[i]            
            data = self.raw_factors[name]
            plt.subplot(rows, 4, i+2)
            plt.plot(dates, data, linewidth = 1.5, color = (0.6, 0.1, 0.1))
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(dates[0], dates[-1])
            plt.grid(True)
            plt.title(name)        
        plt.show()        


    def estimate_parameters(self):
        # estimate the factor loadings
        Lambda_hat = self.V
        # estimate the covariance matrix of the idiosyncratic component
        psi_hat = np.diag(self.S - self.V @ self.D @ self.V.T)
        # estimate the VAR coefficients on the factors
        F_t = self.F[1:, :]
        F_t1 = self.F[:-1, :]
        sum_1 = F_t.T @ F_t1
        sum_2 = F_t1.T @ F_t1
        A_hat = nla.solve(sum_2.T, sum_1.T).T
        # estimate the variance-covariance matrix of factor shocks
        sum_3 = F_t.T @ F_t / (self.T - 1)
        sum_4 = A_hat @ (F_t1.T @ F_t1 / (self.T - 1)) @ A_hat.T
        Sigma_hat = sum_3 - sum_4
        eigenvalues, eigenvectors = nla.eig(Sigma_hat)
        P_hat = np.diag(eigenvalues[:self.q])
        M_hat = eigenvectors[:,:self.q]
        B_hat = M_hat @ np.sqrt(P_hat)
        # save as attributes        
        self.Lambda_hat = Lambda_hat
        self.psi_hat = psi_hat
        self.A_hat = A_hat
        self.B_hat = B_hat
        
        
    def estimate_smoothed_factors(self):
        # prepare observations for the Kalman smoother
        # we don't use a balanced panel anymore, but the full sample with nan
        X = self.processed_features.to_numpy()
        Z = (X - self.X_mean) / self.X_std
        Z_hat = np.zeros((self.nb_periods, self.nb_features))
        F_hat = np.zeros((self.nb_periods, self.r))
        # initiate the Kalman filter
        A = self.A_hat
        Ups = self.B_hat @ self.B_hat.T
        Lbda = self.Lambda_hat
        psi = self.psi_hat
        F_tt = self.F_mean
        Ups_tt = np.diag(3 * self.F_std)
        # run the Kalman filter
        for period in range(self.nb_periods):
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
        X_hat = Z_hat * self.X_std + self.X_mean
        # pass to dataframe
        smoothed_data = pd.DataFrame(index = self.date_list)
        for i in range(self.nb_features):
            smoothed_data[self.feature_list[i]] = X_hat[:,i]
        smoothed_factors = pd.DataFrame(index = self.date_list)
        for i in range(self.r):
            smoothed_factors['factor_' + str(i+1)] = F_hat[:,i]     
        # save as attributes
        self.F_hat = F_hat
        self.Z_hat = Z_hat
        self.X_hat = X_hat
        self.Ups_tt = Ups_tt
        self.smoothed_data = smoothed_data  
        self.smoothed_factors = smoothed_factors        


    def plot_smoothed_factors(self):
        columns = self.r
        rows = columns // 2 + 1
        fig = plt.figure(figsize = (14, 6 * rows))
        plt.suptitle('structural factors: raw and smoothed estimates', 
                     y = 0.95, fontsize = 18, fontweight = 'semibold')
        dates = self.date_list
        for i in range(self.r):            
            data_raw = self.raw_factors['factor_' + str(i+1)]
            data_smoothed = self.smoothed_factors['factor_' + str(i+1)]            
            plt.subplot(rows, 2, i+1)
            plt.plot(dates, data_raw, linewidth = 2, color = (0.1, 0.3, 0.7),
                     label = 'raw factors')
            plt.plot(dates, data_smoothed, linewidth = 2, 
                     color = (0.3, 0.7, 0.1), linestyle = 'dashed', 
                     label = 'smoothed estimates')            
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(dates[0], dates[-1])
            plt.legend()
            plt.grid(True)
            plt.title('factor_' + str(i+1))        
        plt.show()          

    
    def plot_predicted_features(self):
        columns = self.nb_features
        rows = columns // 3 + 1
        fig = plt.figure(figsize = (18, 6 * rows))
        plt.suptitle('features: raw and predicted', 
                     y = 0.90, fontsize = 18, fontweight = 'semibold')
        dates = self.date_list
        for i in range(self.nb_features):           
            data_actual = self.processed_features[self.feature_list[i]]
            data_predicted = self.smoothed_data[self.feature_list[i]]         
            plt.subplot(rows, 3, i+1)
            plt.plot(dates, data_actual, linewidth = 2, color = (0.1, 0.3, 0.7),
                     label = 'actual')
            plt.plot(dates, data_predicted, linewidth = 2, 
                     color = (0.3, 0.7, 0.1), linestyle = 'dashed', 
                     label = 'predicted')            
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(dates[0], dates[-1])
            plt.legend()
            plt.grid(True)
            plt.title(self.feature_list[i])        
        plt.show()
        
        
    def estimate_gdp_prediction_model(self):
        # recover smoothed factors, concatenate gdp, trim months with NaNs
        regression_dataframe = self.smoothed_factors.copy()
        regression_dataframe['quarterly_gdp'] \
                             = self.processed_data['quarterly_gdp']
        regression_dataframe = regression_dataframe.dropna()
        dates = regression_dataframe.index
        # extract regressors and estimate model
        F = np.array(regression_dataframe.iloc[:,:-1])
        F = np.concatenate((np.ones((len(F),1)), F), axis=1)
        y = np.array(regression_dataframe['quarterly_gdp'])
        beta = nla.solve(F.T @ F, F.T @ y)
        # get predictions over sample periods, then keep only quarters
        F = np.array(self.smoothed_factors.copy())
        F = np.concatenate((np.ones((len(F),1)), F), axis=1)
        y_hat = F @ beta
        y_hat = y_hat[self.date_list.month % 3 == 0]
        # pass to dataframe
        for index, date in enumerate(self.gdp_date_list):
            self.smoothed_data.loc[date, 'quarterly_gdp'] = y_hat[index]  
        # save as attributes            
        self.beta = beta
        self.y_hat = y_hat
        
        
    def plot_predicted_gdp(self):
        fig = plt.figure(figsize = (10, 8))
        plt.title('gdp: raw and predicted', 
                     y = 1.02, fontsize = 18, fontweight = 'semibold')
        dates = self.gdp_date_list
        data_actual = self.processed_data.loc[dates, 'quarterly_gdp']
        data_predicted = self.smoothed_data.loc[dates, 'quarterly_gdp']       
        plt.plot(dates, data_actual, linewidth = 2, color = (0.7, 0.1, 0.3), 
                 label = 'actual')
        plt.plot(dates, data_predicted, linewidth = 2, 
                 color = (0.2, 0.6, 0.1), linestyle = 'dashed', 
                 label = 'predicted')            
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
        plt.xlim(dates[0], dates[-1])
        plt.legend()
        plt.grid(True)      
        plt.show()
        
        
    def nowcast_gdp(self, h):
        # extract last month of the sample, and calculate additional months
        year = self.processed_features.index[-1].year
        month = self.processed_features.index[-1].month
        if month == 3 or month == 6 or month == 9 or month == 12:
            months = 0
        elif month == 2 or month == 5 or month == 8 or month == 11:
            months = 1
        elif month == 1 or month == 4 or month == 7 or month == 10:
            months = 2
        if month == 12:
            month = 0
            year += 1
        months += 3 * (h - 1)
        # initiate the Kalman filter to get predictions
        Z_predicted = np.zeros((months, self.nb_features))
        F_predicted = np.zeros((months, self.r))        
        A = self.A_hat
        Ups = self.B_hat @ self.B_hat.T
        Lbda = self.Lambda_hat
        psi = self.psi_hat
        F_tt = self.F_hat[-1,:]
        Ups_tt = self.Ups_tt
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
        X_predicted = Z_predicted * self.X_std + self.X_mean
        F = np.concatenate((np.ones((months,1)), F_predicted), axis=1)
        y_predicted = F @ self.beta
        # pass to dataframe      
        dates = pd.date_range(str(year) + '-' + str(month + 1),
            periods = months, freq='M')
        columns = self.feature_list
        predicted_data = pd.DataFrame(X_predicted, index = dates, 
                                      columns = columns)
        predicted_data['quarterly_gdp'] = y_predicted
        columns = ['factor_' + str(i+1) for i in range(self.r)]
        predicted_factors = pd.DataFrame(F_predicted, index = dates, 
                                         columns = columns)
        smoothed_and_predicted_data = pd.concat([self.smoothed_data, 
                                                 predicted_data])
        smoothed_and_predicted_factors = pd.concat([self.smoothed_factors, 
                                         predicted_factors])
        # calculate specifically the nowcasts for the quarters of interest
        nowcast_quarters = [t+2 for t in range(-3 * h, -1, 3)]
        nowcasts_gdp = smoothed_and_predicted_data.iloc[nowcast_quarters,:] \
                                                            ['quarterly_gdp']
        # save as attributes 
        self.h = h        
        self.F_predicted = F_predicted
        self.Z_predicted = Z_predicted
        self.X_predicted = X_predicted 
        self.y_predicted = y_predicted 
        self.predicted_data = predicted_data
        self.predicted_factors = predicted_factors
        self.smoothed_and_predicted_data = smoothed_and_predicted_data
        self.smoothed_and_predicted_factors = smoothed_and_predicted_factors
        self.nowcasts_gdp = nowcasts_gdp
        
        
    def plot_nowcast_features(self):
        dates_actual = self.date_list
        dataframe_nowcast = pd.concat([self.smoothed_data, 
                                       self.predicted_data])
        dates_nowcast = dataframe_nowcast.index
        columns = self.nb_features
        rows = columns // 3 + 1
        fig = plt.figure(figsize = (18, 6 * rows))
        plt.suptitle('features: raw and nowcast', 
                     y = 0.90, fontsize = 18, fontweight = 'semibold')
        for index, feature in enumerate(self.feature_list):           
            data_actual = self.processed_data[feature]
            data_nowcast = dataframe_nowcast[feature]         
            plt.subplot(rows, 3, index+1)
            plt.plot(dates_actual, data_actual, linewidth = 2,
                     color = (0.1, 0.3, 0.7), label = 'actual')
            plt.plot(dates_nowcast, data_nowcast, linewidth = 2, 
                     color = (0.3, 0.7, 0.1), linestyle = 'dashed', 
                     label = 'predicted')            
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(dates_nowcast[- 3 * self.h - 12], dates_nowcast[-1])
            plt.legend()
            plt.grid(True)
            plt.title(feature)        
        plt.show()
        
          
    def plot_nowcast_gdp(self):
        data_actual = self.processed_data['quarterly_gdp'].dropna()
        dates_actual = data_actual.index
        data_nowcast = self.smoothed_and_predicted_data['quarterly_gdp'] \
                                                        .dropna()
        dates_nowcast = data_nowcast.index
        nowcast_quarters = [t+2 for t in range(-3 * self.h, -1, 3)]
        fig = plt.figure(figsize = (10, 8))
        plt.title('gdp: raw and nowcast', 
                     y = 1.02, fontsize = 18, fontweight = 'semibold')
        plt.plot(dates_actual, data_actual, linewidth = 2, 
                 color = (0.7, 0.1, 0.3), label = 'actual')
        plt.plot(dates_nowcast, data_nowcast, linewidth = 2, 
                 color = (0.2, 0.6, 0.1), linestyle = 'dashed', \
                 label = 'predicted', marker = 'D', markersize = 10 , \
                 markevery = nowcast_quarters)            
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
        plt.xlim(dates_nowcast[- 3 * self.h - 12], dates_nowcast[-1])
        plt.legend()
        plt.grid(True)      
        plt.show()
        for date in dates_nowcast[nowcast_quarters]:
            quarter = date.month / 3
            year = date.year
            value = round(data_nowcast.loc[date], 3)
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        