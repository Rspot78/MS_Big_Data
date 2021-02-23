# imports
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import numpy.random as nrd
import scipy.stats as sst
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
import time



# main class Time_Varying_Bvar
class Time_Varying_Bvar:


    def __init__(self, feature_dataframe):
        features = feature_dataframe
        dates = features.index
        feature_list = features.columns.to_list()
        # get data as array
        y = features.to_numpy()  
        # save as attribute
        self.features = features        
        self.feature_list = feature_list
        self.dates = dates
        self.y = y


    def prior_hyperparameters(self, p):
        # get variables
        y = self.y
        # preliminary OLS estimates
        Y, X, B_hat = self.ols_var(y, p)
        n, T = Y.shape[1], Y.shape[0]
        k = n * p + 1
        Eps = Y - X @ B_hat
        Sigma_hat = Eps.T @ Eps / (T - k - 1)
        inv_Delta_hat, Lambda_hat, _ = sla.ldl(Sigma_hat)
        Delta_hat = nla.inv(inv_Delta_hat)
        # hyperparameters for beta_i
        rho_i = 0.9
        b_i = []
        for i in range(n):
            b_i.append(B_hat[:,i])
        # hyperparameters for Omega_i
        tau = 5
        zeta_0 = k + 5
        Upsilon_0 = 0.01 * np.identity(k)
        # hyperparameters for lambda_i
        gamma_i = 0.9
        s_i = np.diag(Lambda_hat)
        # hyperparameters for phi_i 
        mu = 5
        kappa_0 = 5
        omega_0 = 0.01
        # hyperparameters for delta_i
        alpha_i = 0.9
        d_i = []
        for i in range(n):
            d_i.append(Delta_hat[i,:i])
        # hyperparameters for Psi_i
        epsl = 5
        varphi_0 = []
        Theta_0 = []
        for i in range(n):
            varphi_0.append(i+5)
            Theta_0.append(0.01 * np.identity(i))
        # hyperparameters for r_it (from Kim, Chib and Chephard: Stochastic Volatility: Likelihood Inference and Comparison with ARCH Models)
        m_table = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
        v_table = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])
        q_table = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
        # save as attributes
        self.p = p
        self.Y = Y
        self.X = X
        self.n = n
        self.T = T
        self.B_hat = B_hat
        self.Lambda_hat = np.diag(Lambda_hat)
        self.Delta_hat = Delta_hat
        self.k = k        
        self.rho_i = rho_i
        self.b_i = b_i
        self.tau = tau
        self.zeta_0 = zeta_0
        self.Upsilon_0 = Upsilon_0
        self.gamma_i = gamma_i
        self.s_i = s_i
        self.mu = mu
        self.kappa_0 = 5
        self.omega_0 = 0.01   
        self.alpha_i = alpha_i        
        self.d_i = d_i        
        self.epsl = epsl
        self.varphi_0 = varphi_0
        self.Theta_0 = Theta_0        
        self.m_table = m_table        
        self.v_table = v_table        
        self.q_table = q_table
        
        
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
        
        
    def posterior_hyperparameters(self):
        # unpack
        n, T = self.n, self.T
        rho_i, zeta_0, tau = self.rho_i, self.zeta_0, self.tau
        gamma_i, kappa_0, mu = self.gamma_i, self.kappa_0, self.mu
        alpha_i, varphi_0, epsl = self.alpha_i, self.varphi_0, self.epsl
        v_table, q_table = self.v_table, self.q_table 
        # hyperparameters for Omega_i
        zeta_bar = zeta_0 + T
        F_i = np.identity(T) - rho_i * np.diag(np.ones(T - 1), -1)
        I_tau = np.diag(np.hstack((1/tau, np.ones(T-1))))
        FIF = F_i.T @ I_tau @ F_i
        # hyperparameters for phi_i
        kappa_bar = (T + kappa_0) / 2
        G_i = np.identity(T) - gamma_i * np.diag(np.ones(T - 1), -1)
        I_mu = np.diag(np.hstack((1/mu, np.ones(T-1))))
        GIG = G_i.T @ I_mu @ G_i
        # hyperparameters for Psi_i
        varphi_bar = []
        for i in range(n):
            varphi_bar.append(T + varphi_0[i])
        H_i = np.identity(T) - alpha_i * np.diag(np.ones(T - 1), -1)
        I_epsl = np.diag(np.hstack((1/epsl, np.ones(T-1))))
        HIH = H_i.T @ I_epsl @ H_i
        # hyperparameters for r_it
        c_table = (2 * np.pi * v_table) ** (-1/2) * q_table
        # save as attributes
        self.zeta_bar = zeta_bar
        self.FIF = FIF
        self.kappa_bar = kappa_bar
        self.GIG = GIG
        self.varphi_bar = varphi_bar
        self.HIH = HIH
        self.c_table = c_table


    def initial_values(self):
        # unpack
        n, T, k = self.n, self.T, self.k
        Y, X, B_hat = self.Y, self.X, self.B_hat
        Delta_hat = self.Delta_hat
        m_table, v_table = self.m_table, self.v_table
        # intial values for beta_i, Omega_i, lambda_i, delta_i, and r_i
        beta_i = []
        Omega_i = []
        lambda_i = []
        phi_i = []
        delta_i = []
        Psi_i = []
        r_i = []
        m_i = []
        v_i = []
        y_hat_i = []
        for i in range(n):
            beta_i.append(np.tile(B_hat[:,i].reshape(-1,1),(1,T)))
            Omega_i.append(0.001 * np.identity(k))
            lambda_i.append(np.zeros(T))
            phi_i.append(0.001)
            delta_i.append(np.tile(Delta_hat[i,:i].reshape(-1,1),(1,T)))
            Psi_i.append(0.001 * np.identity(i))
            r_i.append(np.ones(T, dtype = int))
            m_i.append(m_table[r_i[0]])
            v_i.append(v_table[r_i[0]])
            y_hat_i.append(0.001 * np.ones(T))
        # other initial values
        E_i = Y - X @ B_hat
        # save as attributes
        self.beta_i = beta_i
        self.Omega_i = Omega_i
        self.lambda_i = lambda_i
        self.phi_i = phi_i
        self.delta_i = delta_i
        self.Psi_i = Psi_i
        self.r_i = r_i
        self.m_i = m_i
        self.v_i = v_i
        self.y_hat_i = y_hat_i
        self.E_i = E_i
        

    def storage(self, r, d):
        # unpack
        n, T, k = self.n, self.T, self.k
        # generate empty storage lists
        mcmc_beta_i = []
        mcmc_Omega_i = []
        mcmc_lambda_i = []
        mcmc_phi_i = []
        mcmc_delta_i = []
        mcmc_Psi_i = []
        # fill with arrays of zeros to preallocate storage
        for i in range(n):
            mcmc_beta_i.append(np.zeros((k, T, r - d)))
            mcmc_Omega_i.append(np.zeros((k, k, r - d)))
            mcmc_lambda_i.append(np.zeros((1, T, r - d)))
            mcmc_phi_i.append(np.zeros((1, 1, r - d)))
            mcmc_delta_i.append(np.zeros((i, T, r - d)))
            mcmc_Psi_i.append(np.zeros((i, i, r - d)))
        mcmc_Sigma = np.zeros((n * n, T, r - d))
        # save as attributes
        self.r = r
        self.d = d
        self.mcmc_beta_i = mcmc_beta_i
        self.mcmc_Omega_i = mcmc_Omega_i
        self.mcmc_lambda_i = mcmc_lambda_i
        self.mcmc_phi_i = mcmc_phi_i
        self.mcmc_delta_i = mcmc_delta_i
        self.mcmc_Psi_i = mcmc_Psi_i
        self.mcmc_Sigma = mcmc_Sigma


    def mcmc(self, r, d):
        # set initial values and preallocate storage
        self.initial_values()
        self.storage(r, d)
        # loop over iterations and run mcmc algorithm
        t = time.time()
        for i in range(r):
            # draw lambda
            self.mcmc_lambda()            
            # draw beta
            self.mcmc_beta()
            # draw delta
            self.mcmc_delta()
            # draw phi
            self.mcmc_phi()
            # draw Omega
            self.mcmc_Omega()
            # draw Psi
            self.mcmc_Psi()
            # draw r
            self.mcmc_r()
            # display every 10 iteration
            if (i+1) % 10 == 0:
                print('iteration ' + str(i+1))
            # save if burn-in is over
            if i >= d:
                self.store(i, d)
            # save as attributes
            self.r, self.d = r, d
        # display total estimation time (this model can be really long to train)
        elapsed = time.time()-t
        print(str(round(elapsed,2)) + ' second for ' + str(r) + ' iterations.')
                
        
    def mcmc_lambda(self):
        # unpack
        n, y_hat_i, m_i, v_i, gamma_i, phi_i, T, mu = self.n, self.y_hat_i, \
        self.m_i, self.v_i, self.gamma_i, self.phi_i, self.T, self.mu
        # loop over features
        for i in range(n):
            # prepare variables for Kalman filter
            y = y_hat_i[i] - m_i[i]
            Ups = v_i[i]
            C = gamma_i
            K = phi_i[i]
            # prepare initial conditions
            z_t1_t1 = 0
            K_t1_t1 = mu * K
            # Kalman filter, forward pass
            store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t = \
            self.kalman_lambda_forward(y, Ups, C, K, T, z_t1_t1, K_t1_t1)
            # Kalman filter, backward pass
            store_z_t = self.kalman_lambda_backward(C, T, store_z_t_t1, \
            store_z_t_t, store_K_t_t1, store_K_t_t)
            # update lambda_i and values depending on delta_i
            lambda_i = store_z_t
            self.lambda_i[i] = lambda_i


    def kalman_lambda_forward(self, y, Ups, C, K, T, z_t1_t1, K_t1_t1):
        # prepare storage
        store_z_t_t1 = []
        store_z_t_t = []        
        store_K_t_t1 = []
        store_K_t_t = []
        # run the Kalman filter for each period
        for t in range(T):
            # 6 Kalman steps
            z_t_t1 = C * z_t1_t1
            K_t_t1 = C * C * K_t1_t1 + K
            y_t_t1 = z_t_t1
            Ups_t_t1 = K_t_t1 + Ups[t]
            Phi_t = K_t_t1 / Ups_t_t1
            z_t_t = z_t_t1 + Phi_t * (y[t]-y_t_t1)
            K_t_t = K_t_t1 - Phi_t * Ups_t_t1 * Phi_t
            # store for backward recursion
            store_z_t_t1.append(z_t_t1)
            store_z_t_t.append(z_t_t)
            store_K_t_t1.append(K_t_t1)
            store_K_t_t.append(K_t_t)
            # update
            z_t1_t1 = z_t_t
            K_t1_t1 = K_t_t
        return store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t
      
        
    def kalman_lambda_backward(self, C, T, store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t):
        # preallocate storage
        store_z_t = np.zeros(T)
        # sample value for period T
        z_T_T, K_T_T = store_z_t_t[-1], store_K_t_t[-1]
        z_T = nrd.normal(z_T_T, np.sqrt(K_T_T))
        z_t1 = z_T
        store_z_t[-1] = z_T
        # run the filter backward
        for t in range(T - 2, -1, -1):
            # 2 Kalman steps
            z_t_t, z_t1_t = store_z_t_t[t], store_z_t_t1[t + 1]
            K_t_t, K_t1_t = store_K_t_t[t], store_K_t_t1[t + 1]
            Xi_t = C * K_t_t / K_t1_t
            tz_t_t1 = z_t_t + Xi_t * (z_t1 - z_t1_t)
            tK_t_t1 = K_t_t - C * Xi_t * K_t_t
            z_t = nrd.normal(tz_t_t1, np.sqrt(tK_t_t1))
            # store and update
            store_z_t[t] = z_t
            z_t1 = z_t
        return store_z_t        
                
            
    def mcmc_beta(self):
        # unpack
        Y, X, n, beta_i, delta_i, E_i, s_i, lambda_i, rho_i, b_i, Omega_i, T, k, tau = \
        self.Y, self.X, self.n, self.beta_i, self.delta_i, self.E_i, self.s_i, self.lambda_i,\
        self.rho_i, self.b_i, self.Omega_i, self.T, self.k, self.tau
        # loop over features
        for i in range(n):
            # prepare variables for Kalman filter
            y = Y[:,i] + np.diag(E_i[:,:i] @ delta_i[i])
            A = X
            Ups = s_i[i] * np.exp(lambda_i[i])
            w = (1 - rho_i) * b_i[i]
            C = rho_i
            K = Omega_i[i]
            # prepare initial conditions
            z_t1_t1 = b_i[i]
            K_t1_t1 = tau * K
            # Kalman filter, forward pass
            store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t = \
            self.kalman_beta_forward(y, A, Ups, w, C, K, T, z_t1_t1, K_t1_t1)
            # Kalman filter, backward pass
            store_z_t = self.kalman_beta_backward(C, T, k, store_z_t_t1, \
            store_z_t_t, store_K_t_t1, store_K_t_t)
            # update beta_i and values depending on beta_i
            beta_i = store_z_t
            self.beta_i[i] = beta_i           
            self.E_i[:,i] = Y[:,i] - np.diag(X @ beta_i)

            
    def kalman_beta_forward(self, y, A, Ups, w, C, K, T, z_t1_t1, K_t1_t1):
        # prepare storage
        store_z_t_t1 = []
        store_z_t_t = []        
        store_K_t_t1 = []
        store_K_t_t = []
        # run the Kalman filter for each period
        for t in range(T):
            # 6 Kalman steps
            z_t_t1 = w + C * z_t1_t1
            K_t_t1 = C * C * K_t1_t1 + K
            y_t_t1 = A[t,:] @ z_t_t1
            Ups_t_t1 = A[t,:] @ K_t_t1 @ A[t,:] + Ups[t]
            Phi_t = K_t_t1 @ A[t,:] / Ups_t_t1
            z_t_t = z_t_t1 + Phi_t * (y[t]-y_t_t1)
            K_t_t = K_t_t1 - np.outer(Phi_t * Ups_t_t1, Phi_t)
            # store for backward recursion
            store_z_t_t1.append(z_t_t1)
            store_z_t_t.append(z_t_t)
            store_K_t_t1.append(K_t_t1)
            store_K_t_t.append(K_t_t)
            # update
            z_t1_t1 = z_t_t
            K_t1_t1 = K_t_t
        return store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t
                  

    def kalman_beta_backward(self, C, T, k, store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t):
        # preallocate storage
        store_z_t = np.zeros((k, T))
        # sample value for period T
        z_T_T, K_T_T = store_z_t_t[-1], store_K_t_t[-1]
        z_T = nrd.multivariate_normal(z_T_T, K_T_T)
        z_t1 = z_T
        store_z_t[:,-1] = z_T
        # run the filter backward
        for t in range(T - 2, -1, -1):
            # 2 Kalman steps
            z_t_t, z_t1_t = store_z_t_t[t], store_z_t_t1[t + 1]
            K_t_t, K_t1_t = store_K_t_t[t], store_K_t_t1[t + 1]
            Xi_t = C * K_t_t @ nla.inv(K_t1_t)
            tz_t_t1 = z_t_t + Xi_t @ (z_t1 - z_t1_t)
            tK_t_t1 = K_t_t - C * Xi_t @ K_t_t
            z_t = nrd.multivariate_normal(tz_t_t1, tK_t_t1)
            # store and update
            store_z_t[:,t] = z_t
            z_t1 = z_t
        return store_z_t
        
        
    def mcmc_delta(self):
        # unpack
        n, E_i, s_i, lambda_i, alpha_i, d_i, Psi_i, T, epsl = \
        self.n, self.E_i, self.s_i, self.lambda_i,\
        self.alpha_i, self.d_i, self.Psi_i, self.T, self.epsl
        # loop over features
        for i in range(1,n):
            # prepare variables for Kalman filter
            y = E_i[:,i]
            A = - E_i[:,:i]
            Ups = s_i[i] * np.exp(lambda_i[i])
            w = (1 - alpha_i) * d_i[i]
            C = alpha_i
            K = Psi_i[i]
            # prepare initial conditions
            z_t1_t1 = d_i[i]
            K_t1_t1 = epsl * K
            # Kalman filter, forward pass
            store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t = \
            self.kalman_delta_forward(y, A, Ups, w, C, K, T, z_t1_t1, K_t1_t1)
            # Kalman filter, backward pass
            store_z_t = self.kalman_delta_backward(C, T, i, store_z_t_t1, \
            store_z_t_t, store_K_t_t1, store_K_t_t)
            # update delta_i and values depending on delta_i
            delta_i = store_z_t
            self.delta_i[i] = delta_i
            Q_i = (E_i[:,i] + np.diag(E_i[:,:i] @ delta_i)) ** 2
            self.y_hat_i[i] = np.log(Q_i / s_i[i])


    def kalman_delta_forward(self, y, A, Ups, w, C, K, T, z_t1_t1, K_t1_t1):
        # prepare storage
        store_z_t_t1 = []
        store_z_t_t = []        
        store_K_t_t1 = []
        store_K_t_t = []
        # run the Kalman filter for each period
        for t in range(T):
            # 6 Kalman steps
            z_t_t1 = w + C * z_t1_t1
            K_t_t1 = C * C * K_t1_t1 + K
            y_t_t1 = A[t,:] @ z_t_t1
            Ups_t_t1 = A[t,:] @ K_t_t1 @ A[t,:] + Ups[t]
            Phi_t = K_t_t1 @ A[t,:] / Ups_t_t1
            z_t_t = z_t_t1 + Phi_t * (y[t]-y_t_t1)
            K_t_t = K_t_t1 - np.outer(Phi_t * Ups_t_t1, Phi_t)
            # store for backward recursion
            store_z_t_t1.append(z_t_t1)
            store_z_t_t.append(z_t_t)
            store_K_t_t1.append(K_t_t1)
            store_K_t_t.append(K_t_t)
            # update
            z_t1_t1 = z_t_t
            K_t1_t1 = K_t_t
        return store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t


    def kalman_delta_backward(self, C, T, i, store_z_t_t1, store_z_t_t, store_K_t_t1, store_K_t_t):
        # preallocate storage
        store_z_t = np.zeros((i, T))
        # sample value for period T
        z_T_T, K_T_T = store_z_t_t[-1], store_K_t_t[-1]
        z_T = nrd.multivariate_normal(z_T_T, K_T_T)
        z_t1 = z_T
        store_z_t[:,-1] = z_T
        # run the filter backward
        for t in range(T - 2, -1, -1):
            # 2 Kalman steps
            z_t_t, z_t1_t = store_z_t_t[t], store_z_t_t1[t + 1]
            K_t_t, K_t1_t = store_K_t_t[t], store_K_t_t1[t + 1]
            Xi_t = C * K_t_t @ nla.inv(K_t1_t)
            tz_t_t1 = z_t_t + Xi_t @ (z_t1 - z_t1_t)
            tK_t_t1 = K_t_t - C * Xi_t @ K_t_t
            z_t = nrd.multivariate_normal(tz_t_t1, tK_t_t1)
            # store and update
            store_z_t[:,t] = z_t
            z_t1 = z_t
        return store_z_t            


    def mcmc_phi(self):
        # unpack
        n, GIG, kappa_bar, omega_0, lambda_i = self.n, self.GIG, \
        self.kappa_bar, self.omega_0, self.lambda_i
        # loop over features
        for i in range(n):
            # compute Upsilon_bar
            omega_bar = (lambda_i[i] @ GIG @ lambda_i[i] + omega_0) / 2
            # draw phi_i from inverse Gamma and update
            phi_i = sst.invgamma.rvs(kappa_bar, scale = omega_bar)
            self.phi_i[i] = phi_i


    def mcmc_Omega(self):
        # unpack
        n, FIF, zeta_bar, Upsilon_0, beta_i, b_i = self.n, self.FIF, \
        self.zeta_bar, self.Upsilon_0, self.beta_i, self.b_i
        # loop over features
        for i in range(n):
            # compute Upsilon_bar
            temp = beta_i[i].T - b_i[i]
            tB_i = temp.T @ FIF @ temp
            Upsilon_bar = tB_i + Upsilon_0
            # draw Omega_i from inverse Wishart and update
            Omega_i = sst.invwishart.rvs(zeta_bar, Upsilon_bar)
            self.Omega_i[i] = Omega_i
            
            
    def mcmc_Psi(self):
        # unpack
        n, HIH, varphi_bar, Theta_0, delta_i, d_i = self.n, self.HIH, \
        self.varphi_bar, self.Theta_0, self.delta_i, self.d_i
        # loop over features
        for i in range(1, n):
            # compute Theta_bar
            temp = delta_i[i].T - d_i[i]
            tD_i = temp.T @ HIH @ temp
            Theta_bar = tD_i + Theta_0[i]
            # draw Omega_i from inverse Wishart and update
            Psi_i = np.atleast_2d(sst.invwishart.rvs(varphi_bar[i], Theta_bar))
            self.Psi_i[i] = Psi_i


    def mcmc_r(self):
        # unpack
        n, T, c_table, y_hat_i, lambda_i, m_table, v_table = self.n, self.T, \
        self.c_table, self.y_hat_i, self.lambda_i, self.m_table, self.v_table
        # loop over features
        for i in range(n):
            # compute q_bar (up to normalizing constant)
            q_bar = np.exp(-0.5 * np.subtract.outer(y_hat_i[i] - lambda_i[i], m_table) ** 2 / v_table) * c_table
            # normalize to obtain probs summing to 1
            cum_q_bar = np.cumsum(q_bar, 1)
            probs = (cum_q_bar.T / cum_q_bar[:,-1]).T
            # draw from categorical distribution, using posterior probs
            r_i = 7 - np.sum((nrd.rand(T) <= probs.T).T.astype(int), 1)
            # update values
            self.r_i[i] = r_i
            self.m_i[i] = m_table[r_i]
            self.v_i[i] = v_table[r_i]
    
    
    def store(self, iteration, d):
        # unpack
        beta_i, Omega_i, lambda_i, phi_i, delta_i, Psi_i, s_i, n, T = \
        self.beta_i, self.Omega_i, self.lambda_i, self.phi_i, self.delta_i, \
        self.Psi_i, self.s_i, self.n, self.T
        # recover Sigma
        Sigma = self.get_Sigma(delta_i, lambda_i, s_i, n, T)
        self.mcmc_Sigma[:,:,iteration-d] = Sigma
        for i in range(n):
            self.mcmc_beta_i[i][:,:,iteration-d] = beta_i[i]
            self.mcmc_Omega_i[i][:,:,iteration-d] = Omega_i[i]
            self.mcmc_lambda_i[i][:,:,iteration-d] = lambda_i[i]
            self.mcmc_phi_i[i][:,:,iteration-d] = phi_i[i]
            self.mcmc_delta_i[i][:,:,iteration-d] = delta_i[i]
            self.mcmc_Psi_i[i][:,:,iteration-d] = Psi_i[i]
            

    def get_Sigma(self, delta_i, lambda_i, s_i, n, T):
        # initiate Sigma (time-varying)
        Sigma = np.zeros((n * n, T))
        # loop over time periods and variables
        for t in range(T):
            Delta_t = np.identity(n)
            Lambda_t = np.zeros((n, n))
            for i in range(n):
                Delta_t[i,:i] = delta_i[i][:,t]
                Lambda_t[i,i] = s_i[i] * np.exp(lambda_i[i][t])
            inv_Delta_t = nla.inv(Delta_t)
            Sigma_t = inv_Delta_t @ Lambda_t @ inv_Delta_t.T
            Sigma[:,t] = Sigma_t.flatten()
        return Sigma
    
    
    def plot_volatility(self):
        # unpack
        mcmc_Sigma, n, T, r, d, p, feature_list, dates = self.mcmc_Sigma, \
        self.n, self.T, self.r, self.d, self.p, self.feature_list, self.dates
        # get median of posterior distribution
        median_Sigma = np.quantile(mcmc_Sigma, 0.5, 2)
        # retain only the diagonal
        median_volatility = np.zeros((n,T))
        for t in range(T):
            Sigma = np.reshape(median_Sigma[:,t],(n,n))
            median_volatility[:,t] = np.diag(Sigma)
        # dimensionality parameters
        plots_per_row = 3
        rows = n // plots_per_row + np.ceil((n % plots_per_row) / n)
        # plot
        fig = plt.figure(figsize = (18, 4 * rows))
        plt.suptitle('Stochastic volatility of the features', y = 0.93, fontsize = 18, fontweight = 'semibold')
        for i in range(n):
            plt.subplot(rows, plots_per_row, i+1)
            plt.plot(dates[p:], median_volatility[i,:], linewidth = 1.5, color = (0.1, 0.5, 0.1))
            plt.xlim(dates[0], dates[-1])
            plt.title(feature_list[i])
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9))
            plt.grid(True)
        plt.show()
        
        
    def forecast(self, h):
        # unpack
        mcmc_beta_i, mcmc_lambda_i, mcmc_delta_i = self.mcmc_beta_i, self.mcmc_lambda_i, self.mcmc_delta_i
        mcmc_Omega_i,  mcmc_phi_i, mcmc_Psi_i = self.mcmc_Omega_i,  self.mcmc_phi_i, self.mcmc_Psi_i
        rho_i, gamma_i, alpha_i = self.rho_i, self.gamma_i, self.alpha_i
        b_i, s_i, d_i = self.b_i, self.s_i, self.d_i
        n, p, k, T, r, d = self.n, self.p, self.k, self.T, self.r, self.d
        # preallocate storage
        storage_Y_hat = np.zeros((h, n, r - d))
        # predict
        for i in range(r - d):
            # recover parameters
            Y = self.y
            # initiate iteration parameters
            beta_i_t1 = []
            lambda_i_t1 = []
            delta_i_t1 = []
            beta_i_t = [None] * n
            lambda_i_t = [None] * n
            delta_i_t = [None] * n
            Omega_i = []
            phi_i = []
            Psi_i = []
            for j in range(n):
                beta_i_t1.append(mcmc_beta_i[j][:,-1,i])
                lambda_i_t1.append(mcmc_lambda_i[j][:,-1,i])
                delta_i_t1.append(mcmc_delta_i[j][:,-1,i])
                Omega_i.append(mcmc_Omega_i[j][:,:,i])
                phi_i.append(mcmc_phi_i[j][:,:,i])
                Psi_i.append(mcmc_Psi_i[j][:,:,i])
            # loop over periods and update parameters
            for t in range(h):
                # recover regressors
                Y_predict = np.vstack((Y[-p:,:], np.zeros((1,n))))
                y_t, x_t = self.lag_matrix(Y_predict, p)
                # update dynamic parameters
                Lambda_t = np.zeros((n,n))
                Delta_t = np.identity(n)
                for j in range(n):
                    beta_i_t[j] = (1 - rho_i) * b_i[j] + rho_i * beta_i_t1[j] + nrd.multivariate_normal(np.zeros(k), Omega_i[j])
                    lambda_i_t[j] = gamma_i * lambda_i_t1[j] + nrd.normal(0, np.sqrt(phi_i[j]))
                    Lambda_t[j,j] = s_i[j] * np.exp(lambda_i_t[j])
                for j in range(1,n):
                    delta_i_t[j] = (1 - alpha_i) * d_i[j] + alpha_i * delta_i_t1[j] + nrd.multivariate_normal(np.zeros(j), Psi_i[j])
                    Delta_t[j,:j] = delta_i_t[j]
                # recover Sigma_t and draw period t residuals
                inv_Delta_t = nla.inv(Delta_t)
                Sigma_t = inv_Delta_t @ Lambda_t @ inv_Delta_t.T
                eps_t = nrd.multivariate_normal(np.zeros(n), Sigma_t)
                # obtain predictions, equation by equation
                y_hat_t = np.zeros(n)
                for j in range(n):
                    y_hat_t[j] = x_t @ beta_i_t[j] + eps_t[j]
                Y = np.vstack((Y, y_hat_t))
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
        y, n, p, T, h = self.y, self.n, self.p, self.T, self.h
        feature_list, dates = self.feature_list, self.dates
        Y_hat_median = self.Y_hat_median
        Y_hat_lower_bound = self.Y_hat_lower_bound
        Y_hat_upper_bound  = self.Y_hat_upper_bound
        plot_dates = pd.date_range(start = dates[0], periods = T + h, freq ='Q')
        # get plot data for each variable
        plot_values = np.full((T + h, 4, n), np.nan)
        for i in range(n):
            plot_values[:T,0,i] = y[p:,i]
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
      
        
        
        