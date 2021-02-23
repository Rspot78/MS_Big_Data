# note: this is a class to implement the feature transformations proposed in
# Giannone, Reichlin and Small: "Nowcasting: the real-time information content
# of macroeconomic data", Journal of Monetary Economics 55 (2008)


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# class FeatureProcessor
class FeatureProcessor:
    
    
    def __init__(self, path, feature_file, information_file,):
        self.path = path
        self.feature_file = feature_file
        self.information_file = information_file
        
    
    def data_loader(self):
        # unpack
        path = self.path
        feature_file = self.feature_file
        information_file = self.information_file
        # load monthly features and create dataframe
        raw_features = pd.read_csv(path + '/' + feature_file, delimiter=',')
        raw_features = raw_features \
            .set_index(pd.to_datetime(raw_features['date'])).drop('date', 1)
        # load information about monthly data and create dataframe
        information = pd.read_csv(path + '/' + information_file, 
                           delimiter=',', index_col=0)
        # save as attributes
        self.raw_features = raw_features        
        self.information = information        
        
        
    def data_processor(self):
        # unpack
        raw_features = self.raw_features
        information = self.information
        # initiate empty dataframe of processed features
        processed_features = pd.DataFrame(index = raw_features.index.copy())
        # create feature list (omit final entry: quarterly_gdp, left unchanged)
        feature_list = raw_features.columns.to_list()
        # loop over features and implement adequate transformation
        for feature in feature_list[:-1]:
            data = raw_features[feature]
            transformation = information['transformation'] \
            .loc[information['feature'] == feature].tolist()[0]
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
        # concatenate quarterly gdp (no transformation)
        processed_features['quarterly_gdp'] = raw_features['quarterly_gdp'] 
        # trim the first 18 periods used for differencing computations        
        processed_features = processed_features.iloc[18:]
        # supress rows with nan entries in GDP to obtain a balanced sample
        processed_features = processed_features.dropna()
        # create list of feature dates
        feature_dates = processed_features.index
        nb_features, nb_periods = len(feature_list), len(feature_dates)
        # save as attributes
        self.processed_features = processed_features
        self.feature_list = feature_list
        self.feature_dates = feature_dates
        self.nb_features, self.nb_periods = nb_features, nb_periods        
        
        
    def plot_processed_data(self):
        # unpack
        processed_features = self.processed_features
        feature_list = self.feature_list
        feature_dates = self.feature_dates        
        nb_features = self.nb_features
        # plot features
        rows = nb_features // 4 + 1
        dates = feature_dates
        fig = plt.figure(figsize=(18, 4 * rows))
        plt.suptitle('processed features, as Q-to-Q growth rate', 
                     y = 0.95, fontsize = 18, fontweight = 'semibold')
        for i in range(nb_features):
            data = processed_features[feature_list[i]]
            plt.subplot(rows, 4, i+1)
            plt.plot(dates, data, linewidth = 1.5)
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.xlim(dates[0], dates[-1])
            plt.grid(True)
            plt.title(feature_list[i])        
        plt.show()          
        
        

        
        
        
        
        
        
        
        
        
        
        
        