# imports
import numpy as np
import numpy.linalg as nla
import numpy.random as nrd
import scipy.stats as sst
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.patches as mpt
import matplotlib.dates as mdates


# main class Mixed_Frequency_Bvar
class Mixed_Frequency_Bvar:
    
    
    def __init__(self, feature_dataframe):
        features = feature_dataframe
        dates = features.index
        feature_list = features.columns.to_list()
        # split the features into monthly (all but last column which is quarterly GDP)
        monthly_feature_list = feature_list[:-1]
        monthly_features = features.iloc[:,:-1].copy()
        quarterly_feature_list = feature_list[-1]
        quarterly_features = features.iloc[:, [-1]].copy()        
        # create quarterly dataframe with no nan values
        quarterly_features_fill = quarterly_features.copy().fillna(quarterly_features.mean())
        # get variables and dimensions        
        m = len(monthly_feature_list)
        q = len([quarterly_feature_list])
        n = len(feature_list)
        T = len(features)
        # get data as arrays for monthly and quarterly features
        y_m = monthly_features.to_numpy()
        y_q = quarterly_features.to_numpy()
        y_q_fill = quarterly_features_fill.to_numpy()
        # create list of feature arrays, period by period (required for Carter-Kohn algorithm)
        y = []
        for t in range(T):
            y_t = np.concatenate((y_m[t], y_q[t]))
            y_t = y_t[~np.isnan(y_t)]
            y.append(y_t)        
        # save as attribute
        self.features = features        
        self.feature_list = feature_list
        self.dates = dates
        self.monthly_features = monthly_features    
        self.quarterly_features = quarterly_features
        self.quarterly_features_fill = quarterly_features_fill
        self.m = m
        self.q = q
        self.n = n
        self.T = T
        self.y_m = y_m
        self.y_q = y_q
        self.y_q_fill = y_q_fill
        self.y = y        


    def prior(self, p, rho, lambda_1, lambda_2, lambda_3, lambda_4):
        # get dimensions and variables
        m, q, n, T = self.m, self.q, self.n, self.T
        y, y_m, y_q = self.y, self.y_m, self.y_q
        y_q_nonan = y_q[~np.isnan(y_q).any(axis=1)]
        # obtain the individual ar variances (for the Minnesota)
        sigma_m = self.ar_variance(y_m, p)
        sigma_q = self.ar_variance(y_q_nonan, p)
        sigma = np.concatenate((sigma_m, sigma_q))
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
        # generate the series of Lambda_t matrices
        Lambda = []
        for t in range(T):
            if y[t].shape[0] == n:
                Lambda_t = np.hstack((np.identity(n), np.zeros((n, n * (p - 1)))))
            else:
                Lambda_t = np.hstack((np.identity(m), np.zeros((m, q + n * (p - 1)))))
            Lambda.append(Lambda_t)
        # save as attributes
        self.p = p
        self.k = k
        self.inv_Omega_0 = inv_Omega_0
        self.inv_Ombeta_0 = inv_Ombeta_0
        self.nu_0 = nu_0         
        self.S_0 = S_0
        self.Lambda = Lambda        
        

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
        n, k, T, p = self.n, self.k, self.T, self.p
        y_m, y_q_fill = self.y_m, self.y_q_fill
        # set initial values   
        X = np.hstack((y_m, y_q_fill))
        Y, W, B = self.ols_var(X, p)
        Eps = Y - W @ B
        beta = B.flatten('F')
        Sigma = Eps.T @ Eps / T
        # preallocate storage
        storage_X = np.zeros((T, n, r - d))
        storage_beta = np.zeros((n * k, r - d))
        storage_Sigma = np.zeros((n, n, r - d))
        # run algorithm
        for i in range(r): 
            # sample beta
            beta, B = self.mcmc_beta(X, W, Sigma)
            # sample Sigma
            Sigma = self.mcmc_Sigma(X, W, B)
            # sample X
            X, W = self.mcmc_X(beta, Sigma, X)
            if i >= d:
                storage_beta[:,i-d] = beta
                storage_Sigma[:,:,i-d] = Sigma
                storage_X[:,:,i-d] = X
            if (i + 1) % 100 == 0:
                print(str(i + 1) + ' iterations of mcmc algorithm.')
        # save as attribute
        self.r = r
        self.d = d
        self.storage_beta = storage_beta
        self.storage_Sigma = storage_Sigma
        self.storage_X = storage_X


    def mcmc_beta(self, X, W, Sigma):
        # unpack
        k, n, p = self.k, self.n, self.p
        inv_Omega_0, inv_Ombeta_0 = self.inv_Omega_0, self.inv_Ombeta_0
        x = X[p:,:].flatten('F')
        inv_Sigma = nla.inv(Sigma)
        # build posterior elements
        inv_Omega_bar = inv_Omega_0 + np.kron(inv_Sigma, W.T @ W)
        Omega_bar = nla.inv(inv_Omega_bar)
        beta_bar = Omega_bar @ (inv_Ombeta_0 + np.kron(inv_Sigma, W.T) @ x)
        # draw value
        beta = nrd.multivariate_normal(beta_bar, Omega_bar)
        B = np.reshape(beta, (k, n), order='F')
        return beta, B


    def mcmc_Sigma(self, X, W, B):
        # unpack
        p, T = self.p, self.T
        S_0, nu_0 = self.S_0, self.nu_0
        X = X[p:, :]
        # build posterior elements
        Eps = X - W @ B
        S_bar = Eps.T @ Eps + S_0
        nu_bar = T + nu_0
        # draw value
        Sigma = sst.invwishart.rvs(nu_bar, S_bar)
        return Sigma


    def mcmc_X(self, beta, Sigma, X):
        # recover elements for the dynamic equation of z_t
        delta, Phi, Omega = self.companion_form(beta, Sigma)
        # run the Kalman filter, forward pass
        store_z_t_t1, store_z_t_t, store_Om_t_t1, store_Om_t_t = \
                                self.Kalman_forward_pass(X, delta, Phi, Omega)
        # run the Kalman filter, backward pass
        X, W = self.Kalman_backward_pass(Phi, store_z_t_t1, store_z_t_t, \
                                                  store_Om_t_t1, store_Om_t_t)
        return X, W


    def companion_form(self, beta, Sigma):
        k, n, p = self.k, self.n, self.p
        B = np.reshape(beta, (k, n), 'F')
        # compute delta
        delta = np.hstack((B[0,:], np.zeros(((p - 1) * n))))
        # compute Phi
        Phi = np.vstack((B[1:,:].T, np.hstack((np.identity(n * (p-1)), \
                                               np.zeros((n * (p-1), n))))))
        # compute Omega
        Omega = np.zeros((n * p, n * p))
        Omega[:n, :n] = Sigma
        return delta, Phi, Omega
    
    
    def Kalman_forward_pass(self, X, delta, Phi, Omega):
        # unpacking and storage
        Lambda, y, p, n, T = self.Lambda, self.y, self.p, self.n, self.T
        store_z_t_t1 = []
        store_z_t_t = []        
        store_Om_t_t1 = []
        store_Om_t_t = []
        # initial values
        mean = np.mean(X, 0)
        z_t1_t1 = np.tile(mean, p)
        Om_t1_t1 = np.kron(np.identity(p), 5 * Omega[:n, :n])
        # run the Kalman filter from t = 1 to t = T
        for t in range(T):
            Lambda_t = Lambda[t]
            y_t = y[t]
            # 6 Kalman steps
            z_t_t1 = delta + Phi @ z_t1_t1
            Om_t_t1 = Phi @ Om_t1_t1 @ Phi.T + Omega
            y_t_t1 = Lambda_t @ z_t_t1
            Ups_t_t1 = Lambda_t @ Om_t_t1 @ Lambda_t.T
            Psi_t = Om_t_t1 @ Lambda_t.T @ nla.inv(Ups_t_t1)
            z_t_t = z_t_t1 + Psi_t @ (y_t - y_t_t1)
            Om_t_t = Om_t_t1 - Psi_t @ Ups_t_t1 @ Psi_t.T
            # storage for backward pass
            store_z_t_t1.append(z_t_t1)
            store_z_t_t.append(z_t_t)      
            store_Om_t_t1.append(Om_t_t1)
            store_Om_t_t.append(Om_t_t)
            # update for next period
            z_t1_t1 = z_t_t
            Om_t1_t1 = Om_t_t
        return store_z_t_t1, store_z_t_t, store_Om_t_t1, store_Om_t_t


    def Kalman_backward_pass(self, Phi, store_z_t_t1, store_z_t_t, \
                             store_Om_t_t1, store_Om_t_t):
        # unpacking and storage
        p, n, T = self.p, self.n, self.T
        Z = np.zeros((T, n * p))
        # sample period T
        z_T_T, Om_T_T = store_z_t_t[-1], store_Om_t_t[-1]
        z_T = nrd.multivariate_normal(z_T_T, Om_T_T)
        Z[-1,:] = z_T
        z_t1 = z_T
        # backward recursion
        for t in range(T - 2, -1, -1):
            # run the 2 Kalman steps and sample the value
            z_t_t, z_t_t1 = store_z_t_t[t], store_z_t_t1[t + 1]
            Om_t_t, Om_t_t1 = store_Om_t_t[t], store_Om_t_t1[t + 1]
            Xi_t = Om_t_t @ Phi.T @ nla.inv(Om_t_t1)
            tz_t_t1 = z_t_t + Xi_t @ (z_t1 - z_t_t1)
            tOm_t_t1 = Om_t_t - Xi_t @ Phi @ Om_t_t
            z_t = nrd.multivariate_normal(tz_t_t1, tOm_t_t1)
            # store and update
            Z[t,:] = z_t
            z_t1 = z_t
        X = Z[:,:n]
        Y, W = self.create_lag_matrix(X, p)
        return X, W


    def create_lag_matrix(self, A, p):
        n, T = A.shape[1], A.shape[0]
        Y = A[p:, :]
        X = np.ones((T - p, 1))
        for lag in range(p):
            X = np.concatenate((X, A[p-lag-1:T-lag-1, :]), axis = 1)
        return Y, X


    def plot_monthly_gdp(self):
        # plot data
        gdp = self.storage_X[:,-1]
        dates = self.dates
        T = self.T
        median = np.quantile(gdp, 0.5, 1)
        lower_bound = np.quantile(gdp, 0.025, 1)
        upper_bound = np.quantile(gdp, 0.975, 1)
        plot_dates = mdates.date2num(dates)
        # data for monthly gdp, excluding quarters (to avoid strange-looking confidence intervals)
        lower_bound_no_quarter = []
        upper_bound_no_quarter = []
        plot_dates_no_quarter = []
        for t in range(T):
            if not np.allclose(median[t], lower_bound[t]):
                lower_bound_no_quarter.append(lower_bound[t])
                upper_bound_no_quarter.append(upper_bound[t])
                plot_dates_no_quarter.append(plot_dates[t])
        lower_bound_no_quarter = np.array(lower_bound_no_quarter)
        upper_bound_no_quarter = np.array(upper_bound_no_quarter)
        plot_dates_no_quarter = np.array(plot_dates_no_quarter)      
        # patch data
        patch_data = np.hstack((lower_bound_no_quarter, np.flip(upper_bound_no_quarter)))
        patch_dates = np.hstack((plot_dates_no_quarter, np.flip(plot_dates_no_quarter)))
        # space between x ticks
        tick_spacing = 365
        # figure
        fig = plt.figure(figsize=(12,8))
        ax = plt.axes()
        plt.plot(dates, median, linewidth = 2, color = (0.1, 0.3, 0.7))
        years = mdates.YearLocator()
        years_fmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        plt.xlim(dates[0], dates[-1])
        plt.grid(True)
        plt.yticks(np.arange(-12, 8, 2))
        vertices = [*zip(patch_dates, patch_data)]
        poly = mpt.Polygon(vertices, facecolor=[.7, .7, .9], edgecolor=[.8, .8, .9])
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
        plt.ylabel('GDP quarterly growth rate')
        plt.ylim(-3,3)
        ax.add_patch(poly)
        ax.xaxis.set_major_locator(tkr.MultipleLocator(tick_spacing))
        fig.autofmt_xdate()
        plt.title('real GDP growth: monthly estimates, with 95% credibility interval') 
        plt.show();


    def plot_quarterly_gdp(self):
        # plot data
        gdp = self.features.dropna(subset = ["quarterly_gdp"]).iloc[:,-1]
        dates = gdp.index
        # figure
        fig = plt.figure(figsize=(12,8))
        plt.plot(dates, gdp, linewidth = 2, color = (0.1, 0.3, 0.7))
        plt.xlim(dates[0], dates[-1])
        plt.grid(True)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
        plt.ylabel('GDP quarterly growth rate')
        plt.ylim(-3,3)
        plt.title('real GDP growth: actual quarterly data') 
        plt.show();


    def forecast(self, h):
        # unpack
        storage_beta = self.storage_beta
        storage_Sigma = self.storage_Sigma
        storage_X = self.storage_X
        r, d, n, k, p = self.r, self.d, self.n, self.k, self.p
        mean_0 = np.zeros(n)
        # preallocate storage
        storage_X_hat = np.zeros((h, n, r - d))
        # predict
        for i in range(r - d):
            # recover parameters
            B = np.reshape(storage_beta[:,i], (k, n), order='F')
            Sigma = storage_Sigma[:,:,i]
            X = storage_X[:,:,i]
            # loop over forecast periods
            for t in range(h):
                # recover regressors
                X_predict = np.vstack((X[-p:,:], np.zeros((1,n))))
                y, w = self.create_lag_matrix(X_predict, p)
                # create prediction
                Eps = nrd.multivariate_normal(mean_0, Sigma)
                x_hat = w @ B + Eps
                X = np.vstack((X, x_hat))
            X_hat = X[-h:,:]
            storage_X_hat[:,:,i] = X_hat
        # obtain point estimates and credibility bands
        X_hat_median = np.quantile(storage_X_hat, 0.5, 2)
        X_hat_lower_bound = np.quantile(storage_X_hat, 0.15, 2)
        X_hat_upper_bound = np.quantile(storage_X_hat, 0.85, 2)
        # save as attributes
        self.h = h
        self.storage_X_hat = storage_X_hat
        self.X_hat_median = X_hat_median
        self.X_hat_lower_bound = X_hat_lower_bound
        self.X_hat_upper_bound = X_hat_upper_bound 


    def plot_forecast(self):
        # unpack elements for plots
        n, T, h = self.n, self.T, self.h
        feature_list, dates = self.feature_list, self.dates
        y_m, y_q = self.y_m, self.y_q    
        y = np.hstack((self.y_m, self.y_q))
        X_hat_median = self.X_hat_median
        X_hat_lower_bound = self.X_hat_lower_bound
        X_hat_upper_bound  = self.X_hat_upper_bound
        plot_dates = pd.date_range(start = dates[0], periods = T + h, freq ='M')
        # get plot data for each variable
        plot_values = np.full((T + h, 4, n), np.nan)
        for i in range(n):
            plot_values[:T,0,i] = y[:,i]
            plot_values[T-1,1:4,i] = y[-1,i]
            plot_values[T-2,1:4,i] = y[-2,i]
            plot_values[T-3,1:4,i] = y[-3,i]
            plot_values[T:,1,i] = X_hat_lower_bound[:,i]
            plot_values[T:,2,i] = X_hat_median[:,i]
            plot_values[T:,3,i] = X_hat_upper_bound[:,i]
        # plot, looping over variables:
        fig = plt.figure(figsize=(16,18))
        plt.suptitle('Feature predictions with 95% credibility intervals', \
                     y=0.93, fontsize=18, fontweight='semibold')
        rows, columns = n // 3 + 1, 3     
        for i in range(n):
            # remove all-nan rows and plot
            ax = plt.subplot(rows, columns, i+1)
            values_0 = plot_values[:,0,i]
            mask_0 = np.isnan(values_0)            
            plt.plot(plot_dates[~mask_0], values_0[~mask_0], linewidth = 2, color = (0.1, 0.3, 0.7))
            values_2 = plot_values[:,2,i]
            mask_2 = np.isnan(values_2)            
            plt.plot(plot_dates[~mask_2], values_2[~mask_2], linewidth = 2, color = (0.2, 0.6, 0.2))
            # remove all-nan rows and patch         
            values_1 = plot_values[:,1,i]
            mask_1 = np.isnan(values_1)
            plot_dates_1 = np.array(mdates.date2num(plot_dates[~mask_1]))            
            values_3 = plot_values[:,3,i]
            mask_3 = np.isnan(values_3)
            plot_dates_3 = np.array(mdates.date2num(plot_dates[~mask_3]))
            patch_data = np.hstack((values_1[~mask_1], np.flip(values_3[~mask_3])))
            patch_dates = np.hstack((plot_dates_1, np.flip(plot_dates_1)))
            vertices = [*zip(patch_dates, patch_data)]
            poly = mpt.Polygon(vertices, facecolor=[.7, .9, .7], edgecolor=[.5, .7, .5])
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))            
            ax.add_patch(poly)
            plt.title(feature_list[i]) 
            plt.xlim(plot_dates[0], plot_dates[-1])
            plt.grid(True)
        plt.show()

















