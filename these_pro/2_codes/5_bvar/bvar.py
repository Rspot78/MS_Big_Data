# imports
import numpy as np
import numpy.linalg as nla
import numpy.random as nrd
import scipy.stats as sst
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr



# main class Bayesian_Var
class Bayesian_Var:


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


    def prior(self, p, rho, lambda_1, lambda_2, lambda_3, lambda_4):
        # get dimensions and variables
        y, n, T = self.y, self.n, self.T
        # obtain the individual ar variances (for the Minnesota)
        sigma = self.ar_variance(y, p)
        # calculate the number of coefficients for one equation in the VAR
        k = n * p + 1
        # generate beta_0
        beta_0 = np.concatenate((np.zeros((1,n)), rho *  np.identity(n), \
                                    np.zeros((n * (p - 1), n)))).flatten('F')        
        # generate omega_0
        omega_0 = np.zeros((k, n))
        for i in range(n):
            temp_1 = lambda_1 * np.ones(k)
            temp_2 = np.hstack(([1], np.tile(np.hstack((lambda_2 * np.ones(i), \
                     [1], lambda_2 * np.ones(n - i - 1))), p)))
            temp_3 = np.hstack(([1], np.kron(np.arange(1,p+1), np.ones(n)))) ** lambda_3
            temp_4 = np.hstack(([lambda_4], np.ones(k - 1)))
            temp_5 = sigma[i] / np.hstack(([1], np.tile(sigma, p)))
            omega_0[:, i] = (temp_1 * temp_2 / temp_3 * temp_4) ** 2 * temp_5
        omega_0 = omega_0.flatten('F')
        inv_omega_0 = 1 / omega_0
        inv_Omega_0 = np.diag(inv_omega_0)
        inv_Ombeta_0 = inv_omega_0 * beta_0
        # generate nu_0
        nu_0 = n + 2
        # generate S_0
        S_0 = np.diag(sigma)
        # save as attributes
        self.p = p
        self.k = k
        self.inv_Omega_0 = inv_Omega_0
        self.inv_Ombeta_0 = inv_Ombeta_0
        self.nu_0 = nu_0         
        self.S_0 = S_0


    def ar_variance(self, Y, p):
        n = Y.shape[1]
        sigma = np.zeros((n))
        for i in range(n):
            y = Y[:, i].reshape((-1, 1))
            y, X, beta = self.ols_var(y, p)
            eps = y - X @ beta
            sigma[i] = np.var(eps)
        return sigma


    def ols_var(self, Y, p):
        Y, X = self.lag_matrix(Y, p)
        B = nla.solve(X.T @ X, X.T @ Y)
        return Y, X, B


    def lag_matrix(self, A, p):
        n, T = A.shape[1], A.shape[0]
        Y = A[p:, :]
        X = np.ones((T - p, 1))
        for lag in range(p):
            X = np.concatenate((X, A[p-lag-1:T-lag-1, :]), axis = 1)
        return Y, X


    def mcmc(self, r, d):
        # unpack
        y, n, k, T, p = self.y, self.n, self.k, self.T, self.p
        # set initial values   
        Y, X, B = self.ols_var(y, p)
        Eps = Y - X @ B
        beta = B.flatten('F')
        Sigma = Eps.T @ Eps / T
        # preallocate storage
        storage_beta = np.zeros((n * k, r - d))
        storage_Sigma = np.zeros((n, n, r - d))
        # run algorithm
        for i in range(r): 
            # sample beta
            beta, B = self.mcmc_beta(Y, X, Sigma)
            # sample Sigma
            Sigma = self.mcmc_Sigma(Y, X, B)
            if i >= d:
                storage_beta[:,i-d] = beta
                storage_Sigma[:,:,i-d] = Sigma
            if (i + 1) % 100 == 0:
                print(str(i + 1) + ' iterations of mcmc algorithm.')
        # save as attribute
        self.r = r
        self.d = d
        self.storage_beta = storage_beta
        self.storage_Sigma = storage_Sigma


    def mcmc_beta(self, Y, X, Sigma):
        # unpack
        k, n, p = self.k, self.n, self.p
        inv_Omega_0, inv_Ombeta_0 = self.inv_Omega_0, self.inv_Ombeta_0
        y = Y.flatten('F')
        inv_Sigma = nla.inv(Sigma)
        # build posterior elements
        inv_Omega_bar = inv_Omega_0 + np.kron(inv_Sigma, X.T @ X)
        Omega_bar = nla.inv(inv_Omega_bar)
        beta_bar = Omega_bar @ (inv_Ombeta_0 + np.kron(inv_Sigma, X.T) @ y)
        # draw value
        beta = nrd.multivariate_normal(beta_bar, Omega_bar)
        B = np.reshape(beta, (k, n), order='F')
        return beta, B


    def mcmc_Sigma(self, Y, X, B):
        # unpack
        p, T = self.p, self.T
        S_0, nu_0 = self.S_0, self.nu_0
        # build posterior elements
        Eps = Y - X @ B
        S_bar = Eps.T @ Eps + S_0
        nu_bar = T + nu_0
        # draw value
        Sigma = sst.invwishart.rvs(nu_bar, S_bar)
        return Sigma


    def forecast(self, h):
        # unpack
        storage_beta = self.storage_beta
        storage_Sigma = self.storage_Sigma
        r, d, n, k, p = self.r, self.d, self.n, self.k, self.p
        mean_0 = np.zeros(n)
        # preallocate storage
        storage_Y_hat = np.zeros((h, n, r - d))
        # predict
        for i in range(r - d):
            # recover parameters
            Y = self.y
            B = np.reshape(storage_beta[:,i], (k, n), order='F')
            Sigma = storage_Sigma[:,:,i]
            # loop over forecast periods
            for t in range(h):
                # recover regressors
                Y_predict = np.vstack((Y[-p:,:], np.zeros((1,n))))
                y, x = self.lag_matrix(Y_predict, p)
                # create prediction
                Eps = nrd.multivariate_normal(mean_0, Sigma)
                y_hat = x @ B + Eps
                Y = np.vstack((Y, y_hat))
            Y_hat = Y[-h:,:]
            storage_Y_hat[:,:,i] = Y_hat
        # obtain point estimates and credibility bands
        Y_hat_median = np.quantile(storage_Y_hat, 0.5, 2)
        Y_hat_lower_bound = np.quantile(storage_Y_hat, 0.15, 2)
        Y_hat_upper_bound = np.quantile(storage_Y_hat, 0.85, 2)
        # save as attributes
        self.h = h
        self.storage_Y_hat = storage_Y_hat
        self.Y_hat_median = Y_hat_median
        self.Y_hat_lower_bound = Y_hat_lower_bound
        self.Y_hat_upper_bound = Y_hat_upper_bound 


    def plot_forecast(self):
        # unpack elements for plots
        y, n, T, h = self.y, self.n, self.T, self.h
        feature_list, dates = self.feature_list, self.dates
        Y_hat_median = self.Y_hat_median
        Y_hat_lower_bound = self.Y_hat_lower_bound
        Y_hat_upper_bound  = self.Y_hat_upper_bound
        plot_dates = pd.date_range(start = dates[0], periods = T + h, freq ='Q')
        # get plot data for each variable
        plot_values = np.full((T + h, 4, n), np.nan)
        for i in range(n):
            plot_values[:T,0,i] = y[:,i]
            plot_values[T-1,1:4,i] = y[-1,i]
            plot_values[T-2,1:4,i] = y[-2,i]
            plot_values[T-3,1:4,i] = y[-3,i]
            plot_values[T:,1,i] = Y_hat_lower_bound[:,i]
            plot_values[T:,2,i] = Y_hat_median[:,i]
            plot_values[T:,3,i] = Y_hat_upper_bound[:,i]
        # plot, looping over variables:
        fig = plt.figure(figsize=(16,18))
        plt.suptitle('Feature predictions with 95% credibility intervals', \
                     y=0.93, fontsize=18, fontweight='semibold')
        rows, columns = n // 3 + 1, 3     
        for i in range(n):
            # remove all-nan rows and plot
            ax = plt.subplot(rows, columns, i+1)           
            plt.plot(plot_dates, plot_values[:,0,i], linewidth = 2, color = (0.1, 0.3, 0.7))
            plt.plot(plot_dates, plot_values[:,2,i], linewidth = 2, color = (0.2, 0.6, 0.2))
            # remove all-nan rows and patch         
            values_1 = plot_values[:,1,i]
            mask_1 = np.isnan(values_1)
            plot_dates_1 = np.array(mdates.date2num(plot_dates[~mask_1]))
            values_3 = plot_values[:,3,i]
            mask_3 = np.isnan(values_3)
            plot_dates_3 = np.array(mdates.date2num(plot_dates[~mask_3]))
            patch_data = np.hstack((values_1[~mask_1], np.flip(values_3[~mask_3])))
            patch_dates = np.hstack((plot_dates_1, np.flip(plot_dates_3)))
            vertices = [*zip(patch_dates, patch_data)]
            poly = mpt.Polygon(vertices, facecolor=[.7, .9, .7], edgecolor=[.5, .7, .5])
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))            
            ax.add_patch(poly)
            plt.title(feature_list[i]) 
            plt.xlim(plot_dates[0], plot_dates[-1])
            plt.grid(True)
        plt.show()










