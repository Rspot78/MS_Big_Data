# imports
from dynamic_factor_model import *
import numpy as np
import pandas as pd


# main class DynamicFactorIterator
class DynamicFactorIterator:
    
    
    def __init__(self):
        pass
    
    
    def generate_date_list(self, rolling_window, rolling_window_end):
        year = int(rolling_window_end.split('m')[0])
        month = int(rolling_window_end.split('m')[1])
        date_list = []
        for i in range(rolling_window):
            date_list.append(str(year) + 'm' + str(month))
            month -= 1
            if month == 0:
                year -= 1
                month = 12
        date_list.reverse()
        return date_list
    
    
    def month_to_last_quarter(self, date):
        year, month = int(date.split('m')[0]), int(date.split('m')[1])
        quarter = (month - 1) // 3
        if quarter == 0:
            quarter = 4
            year -= 1
        return str(year) + 'q' + str(quarter)

            
    def hyperparameter_optimizer(self, path, feature_file, information_file, 
                                gdp_file, factors, shocks, rolling_window, 
                                rolling_window_end, h):
        # create dataframe of full data
        full_features = pd.read_csv(path + '/' + feature_file, delimiter=',') \
                                                        .set_index('date')
        full_gdp = pd.read_csv(path + '/' + gdp_file, delimiter=',') \
                                                        .set_index('date')
        # prepare array to store rmse values
        storage_rmse = np.zeros((len(factors), len(shocks), rolling_window))
        # create list of dates to iterate
        dates = self.generate_date_list(rolling_window, rolling_window_end)
        # loop over factors and shocks
        for row, factor in enumerate(factors):
            for column, shock in enumerate(shocks):
                # loop over rolling sample
                for page, date in enumerate(dates):
                    # trim features at rolling sample limit and write to csv
                    rolling_features = full_features.loc[:date, :]
                    rolling_features.to_csv(path + '/rolling_features.csv')
                    # trim gdp at rolling sample limit and write to csv
                    quarterly_date = self.month_to_last_quarter(date)
                    rolling_gdp = full_gdp.loc[:quarterly_date, :]
                    rolling_gdp.to_csv(path + '/rolling_gdp.csv')
                    # create dynamic factor model, train and predict
                    dfm = DynamicFactorModel(factor, shock)
                    dfm.load(path, 'rolling_features.csv', information_file, 
                             'rolling_gdp.csv')
                    dfm.train()
                    dfm.predict(h)
                    # record final prediction
                    prediction = dfm.nowcasts_gdp.to_list()[-1]
                    # recover actual value
                    last_quarter_index = full_gdp.index.get_loc(quarterly_date)
                    prediction_quarter_index = last_quarter_index + h
                    actual = full_gdp['gdp'].iloc[prediction_quarter_index]
                    # compute RMSE and record
                    rmse = np.sqrt((prediction - actual) ** 2)
                    storage_rmse[row, column, page] = rmse
        # average RMSE over rolling window periods
        mean_rmse = np.mean(storage_rmse, axis = 2)
        # find factors and shocks for min rmse
        optimal_indices = np.argwhere(mean_rmse == np.min(mean_rmse))
        optimal_factor = optimal_indices[0][0]
        optimal_shocks = optimal_indices[0][1]
        print('The optimal number of factors is: ' + str(optimal_factor) + '.')
        print('The optimal number of shocks is: ' + str(optimal_shocks) + '.')
        # save as attributes
        self.storage_rmse_optimize = storage_rmse
        self.mean_rmse_optimize = mean_rmse
        self.optimal_factor = optimal_factor
        self.optimal_shocks = optimal_shocks
        

    def gdp_prediction_evaluator(self, path, feature_file, information_file, 
                                gdp_file, factors, shocks, rolling_window, 
                                rolling_window_end, h):    
        # create dataframe of full data
        full_features = pd.read_csv(path + '/' + feature_file, delimiter=',') \
                                                        .set_index('date')
        full_gdp = pd.read_csv(path + '/' + gdp_file, delimiter=',') \
                                                        .set_index('date')
        # prepare arrays to store prediction and rmse values
        storage_prediction = np.zeros((rolling_window, h))
        storage_rmse = np.zeros((rolling_window, h))
        # create list of dates to iterate
        dates = self.generate_date_list(rolling_window, rolling_window_end)    
        for row, date in enumerate(dates):
            # trim features at rolling sample limit and write to csv
            rolling_features = full_features.loc[:date, :]
            rolling_features.to_csv(path + '/rolling_features.csv')
            # trim gdp at rolling sample limit and write to csv
            quarterly_date = self.month_to_last_quarter(date)
            rolling_gdp = full_gdp.loc[:quarterly_date, :]
            rolling_gdp.to_csv(path + '/rolling_gdp.csv')
            # create dynamic factor model, train and predict
            dfm = DynamicFactorModel(factors, shocks)
            dfm.load(path, 'rolling_features.csv', information_file, 
                     'rolling_gdp.csv')
            dfm.train()
            dfm.predict(h)
            # recover prediction and record
            prediction = np.array(dfm.nowcasts_gdp.to_list())
            storage_prediction[row, :] = prediction
            # recover actual value
            last_quarter_index = full_gdp.index.get_loc(quarterly_date)
            actual = np.array(full_gdp['gdp'] \
                .iloc[last_quarter_index + 1 : last_quarter_index + h + 1] \
                .to_list())
            # compute RMSE and record
            rmse = np.sqrt((prediction - actual) ** 2)
            storage_rmse[row, :] = rmse    
            # average RMSE over rolling window periods
        mean_rmse = np.mean(storage_rmse, axis = 0)
        mean_rmse_dataframe = pd.DataFrame(mean_rmse, index = range(1, h + 1),\
                         columns = ['quarterly_gdp'])
        for i in range(h):
            print('The mean RMSE for the prediction ' + str(i + 1) \
                  + ' quarter ahead is ' + str(round(mean_rmse[i], 3)) + '.')
        # save as attributes
        self.storage_prediction_gdp = storage_prediction
        self.storage_rmse_gdp = storage_rmse
        self.mean_rmse_gdp = mean_rmse_dataframe
    
    
    def feature_prediction_evaluator(self, path, feature_file, information_file, 
                                gdp_file, factors, shocks, rolling_window, 
                                rolling_window_end, h):    
        # create dataframe of full data
        full_features = pd.read_csv(path + '/' + feature_file, delimiter=',') \
                                                        .set_index('date')
        full_gdp = pd.read_csv(path + '/' + gdp_file, delimiter=',') \
                                                        .set_index('date')
        # process features to stationarize them
        dfm = DynamicFactorModel()
        dfm.load_data(path, feature_file, information_file, gdp_file)
        dfm.process_data()
        full_processed_features = dfm.processed_data.iloc[:,:-1]
        # prepare arrays to store prediction and rmse values
        storage_prediction = np.zeros((h, full_processed_features.shape[1] \
                                                             ,rolling_window))
        storage_rmse = np.zeros((h, full_processed_features.shape[1] \
                                                             ,rolling_window))
        # create list of dates to iterate
        dates = self.generate_date_list(rolling_window, rolling_window_end)    
        for page, date in enumerate(dates):
            # trim features at rolling sample limit and write to csv
            rolling_features = full_features.loc[:date, :]
            rolling_features.to_csv(path + '/rolling_features.csv')
            # trim gdp at rolling sample limit and write to csv
            quarterly_date = self.month_to_last_quarter(date)
            rolling_gdp = full_gdp.loc[:quarterly_date, :]
            rolling_gdp.to_csv(path + '/rolling_gdp.csv')
            # create dynamic factor model, train and predict
            dfm = DynamicFactorModel(factors, shocks)
            dfm.load(path, 'rolling_features.csv', information_file, 
                     'rolling_gdp.csv')
            dfm.train()
            dfm.predict(h)
            # recover prediction and record
            nowcast_index = [t+2 for t in range(-3 * h, -1, 3)]
            prediction = dfm.smoothed_and_predicted_data \
                                        .iloc[nowcast_index,:-1]
            storage_prediction[:, :, page] = prediction
            # recover actual value
            actual = full_processed_features.loc[prediction.index]
            # compute RMSE and record
            rmse = np.sqrt((np.array(prediction) - np.array(actual)) ** 2)
            storage_rmse[:, :, page] = rmse
            # average RMSE over rolling window periods
        mean_rmse = np.mean(storage_rmse, axis = 2)
        mean_rmse_dataframe = pd.DataFrame(mean_rmse, index = range(1, h + 1),\
                                 columns = full_processed_features.columns)
        # save as attributes
        self.storage_prediction_features = storage_prediction
        self.storage_rmse_features = storage_rmse
        self.mean_rmse_features = mean_rmse_dataframe  
    
    
    
    
    
    
