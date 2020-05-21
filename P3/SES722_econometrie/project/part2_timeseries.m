% preliminaries
clear
clc
close all
% cd 'path_to_data'; % if needed, to change working directory



%--------------------------
% Part 2: time-series     |
%--------------------------


% question 1

data = xlsread("quarterly.xls");
[nobs, nvar] = size(data);
nb_nan = sum(sum(isnan(data)));
disp('question 1')
disp(' ')
disp(['number of variables in the quarterly dataset: ' num2str(nvar)])
disp(['number of observations in the quarterly dataset: ' num2str(nobs)])
disp(['number of missing observations: ' num2str(nb_nan)])
disp("The are no missing observations, hence no need for pre-treatment of the dataset.")
disp(' ')
disp(' ')


% question 2

cpi = data(:, 8);
time = (1960:0.25:2012.75)';
fig_2_1 = figure;
set(fig_2_1, 'color', [.92 .92 .90], "units", "pixels", "position", [300, 300, 600, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 2: plot of CPI")
plot(time, cpi, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
xlabel('time periods')
ylabel('CPI')
xlim([1960 2012.75])
ylim([0 250])
title('question 2, figure 1: plot of CPI', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'normal');
print('-clipboard', '-dbitmap')
trend = (1:nobs)';
trend2 = trend.^2;
trendlog = log(trend);
cpilog = log(cpi);
const = ones(numel(trend),1);
X_t1 = [const trend];
beta_t1 = (X_t1'*X_t1)\(X_t1'*cpi);
t1 = X_t1 * beta_t1;
cpi_t1 = cpi - t1;
X_t2 = [const trend2];
beta_t2 = (X_t2'*X_t2)\(X_t2'*cpi);
t2 = X_t2 * beta_t2;
cpi_t2 = cpi - t2;
X_t3 = [const trendlog];
beta_t3 = (X_t3'*X_t3)\(X_t3'*cpi);
t3 = X_t3 * beta_t3;
cpi_t3 = cpi - t3;
X_t4 = X_t1;
beta_t4 = (X_t4'*X_t4)\(X_t4'*cpilog);
t4 = exp(X_t4 * beta_t4);
cpi_t4 = cpi - t4;
fig_2_2 = figure;
set(fig_2_2, 'color', [.92 .92 .90], "units", "pixels", "position", [300, 300, 1200, 500], "MenuBar", "none", "NumberTitle", "off", "name", "question 2: plots of trends")
subplot(2, 4, 1)
hold on
plot(time, cpi, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
plot(time, t1, '--', 'LineWidth', 2, 'Color', [0.8 0.2 0.3]);
hold off
xlim([1960 2012.75])
title('cpi (linear trend)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
subplot(2, 4, 5)
hold on
plot(time, cpi_t1, 'LineWidth', 2, 'Color', [0.3 0.7 0.2])
hold off
xlim([1960 2012.75])
title('detrended cpi (linear)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
subplot(2, 4, 2)
hold on
plot(time, cpi, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
plot(time, t2, '--', 'LineWidth', 2, 'Color', [0.8 0.2 0.3]);
hold off
xlim([1960 2012.75])
title('cpi (quadratic trend)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
subplot(2, 4, 6)
hold on
plot(time, cpi_t2, 'LineWidth', 2, 'Color', [0.3 0.7 0.2])
hold off
xlim([1960 2012.75])
title('detrended cpi (quadratic)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
subplot(2, 4, 3)
hold on
plot(time, cpi, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
plot(time, t3, '--', 'LineWidth', 2, 'Color', [0.8 0.2 0.3]);
hold off
xlim([1960 2012.75])
title('cpi (log trend)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
subplot(2, 4, 7)
hold on
plot(time, cpi_t3, 'LineWidth', 2, 'Color', [0.3 0.7 0.2])
hold off
xlim([1960 2012.75])
title('detrended cpi (log)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
subplot(2, 4, 4)
hold on
plot(time, cpi, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
plot(time, t4, '--', 'LineWidth', 2, 'Color', [0.8 0.2 0.3]);
hold off
xlim([1960 2012.75])
title('cpi (exp trend)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
subplot(2, 4, 8)
hold on
plot(time, cpi_t4, 'LineWidth', 2, 'Color', [0.3 0.7 0.2])
hold off
xlim([1960 2012.75])
title('detrended cpi (exp)', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
sgtitle('question 2, figure 2: plots of different trends for the cpi series')
print('-clipboard', '-dbitmap')
disp('question 2')
disp(' ')
disp("please refer to figures 1 and 2")
disp(' ')
disp(' ')


% question 3

mm_dim = 5;
t5 = zeros(nobs,1);
for ii = mm_dim:nobs-mm_dim+1
    summ = 0;
    summ = summ + 1/mm_dim * cpi(ii);
    for jj = 1:(mm_dim-1)
        summ = summ + (mm_dim-jj)/(mm_dim^2) * cpi(ii-jj) + (mm_dim-jj)/(mm_dim^2) * cpi(ii+jj);
    end
    t5(ii)=summ;
end
cpi_t5 = cpi - t5;
t5(1:mm_dim-1) = nan;
cpi_t5(1:mm_dim-1) = nan; 
t5(nobs-mm_dim+2:end) = nan;
cpi_t5(nobs-mm_dim+2:end) = nan; 
fig_3_1 = figure;
set(fig_3_1, 'color', [.92 .92 .90], "units", "pixels", "position", [300, 300, 1200, 300], "MenuBar", "none", "NumberTitle", "off", "name", "question 3: plots of cpi: actual, trend and detrended")
subplot(1, 3, 1)
plot(time, cpi, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
xlabel('time periods')
ylabel('CPI')
xlim([1960 2012.75])
subplot(1, 3, 2)
plot(time, t5, '--', 'LineWidth', 2, 'Color', [0.8 0.2 0.3])
xlabel('time periods')
ylabel('moving average trend')
xlim([1960 2012.75])
subplot(1, 3, 3)
plot(time, cpi_t5, 'LineWidth', 2, 'Color', [0.3 0.7 0.2])
xlabel('time periods')
ylabel('detrended CPI')
xlim([1960 2012.75])
sgtitle('question 3, figure 3: plots of cpi: actual, trend and detrended with moving average(5,5)')
print('-clipboard', '-dbitmap')
disp('question 3')
disp(' ')
disp("please refer to figure 3")
disp(' ')
disp(' ')


% question 4

inf = [nan; 400*(cpilog(2:end)-cpilog(1:end-1))];
fig_4_1 = figure;
set(fig_4_1, 'color', [.92 .92 .90], "units", "pixels", "position", [300, 300, 600, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 4: plot of Inflation")
plot(time, inf, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
xlabel('time periods')
ylabel('Inflation')
xlim([1960 2012.75])
ylim([-10 20])
title('question 4, figure 4: plot of Inflation', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'normal');
print('-clipboard', '-dbitmap')
disp('question 4')
disp(' ')
disp("please refer to figure 4")
disp(' ')
disp(' ')


% question 5

fig_5_1 = figure;
set(fig_5_1, 'color', [.92 .92 .90], "units", "pixels", "position", [300, 300, 1200, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 5: ACF and PACF of Inflation")
subplot(1, 2, 1)
autocorr(inf)
subplot(1, 2, 2)
parcorr(inf)
sgtitle('question 5, figure 5: ACF and PACF of Inflation')
print('-clipboard', '-dbitmap')
disp('question 5')
disp(' ')
disp("please refer to figure 5")
disp(' ')
disp(' ')


% question 6

disp('question 6')
disp(' ')
disp("please refer to question 6 in the report")
disp(' ')
disp(' ')


% question 7

disp('question 7')
disp(' ')
[h,~,~,~,reg] = adftest(inf,'model','ARD','lags',1:5);
[aic,nlags] = min([reg.AIC]);
disp(["The minimum AIC value for the ADF test is " + num2str(aic) + ", which corresponds to " + num2str(nlags) + " lags"])
[~,pval,~,~,~] = adftest(inf,'model','ARD','lags', nlags);
if pval<0.05
    disp("the p-value for the ADF test with drift is " + num2str(pval)) 
    disp("this is smaller than the 5% threshold, hence reject the null and conclude that the series is stationary")
else
    disp("the p-value for the ADF test with drift is " + num2str(pval)) 
    disp("this is larger than the 5% threshold, hence do not reject the null and conclude that the series is non-stationary")
end
disp(' ')
disp(' ')


% question 8

AIC = [];
BIC = [];
inf = inf(2:end);
for p=1:10
    T = numel(inf) - p;
    y = inf(p+1:end);
    X = ones(T,1);
    for lag=1:p
        X = [X inf(p+1-lag:T+p-lag)];
    end
    rho = (X'*X)\(X'*y);
    res = y-X*rho;
    sig2 = 1/(T-numel(rho))*res'*res;
    AIC = [AIC; log(sig2)+ 2*p/T];
    BIC = [BIC; log(sig2)+ log(T)*2*p/T];
end     
[aic,nlagsaic] = min(AIC);
[bic,nlagsbic] = min(BIC);
results = [" " "AIC" "BIC";(1:10)' AIC BIC];
disp('question 8')
disp(' ')
disp("The results for the AIC and BIC information criteria are:")
disp(results)
disp("The minimum value for AIC is " + num2str(aic) + ", which corresponds to " + num2str(nlagsaic) + " lags")
disp("The minimum value for BIC is " + num2str(bic) + ", which corresponds to " + num2str(nlagsbic) + " lags")
disp(' ')
disp(' ')


% question 9

unemp = data(2:end,13);
T = numel(unemp);
y = unemp;
X = [ones(T,1) inf];
k = 2;
beta = (X'*X)\(X'*y);
u = y - X*beta;
s2 = u'*u/(T-k);
S = (X'*X)\eye(k);
stderr = diag(s2*S).^0.5;
tstat = abs(beta./stderr);
critvals = [tinv(0.95,T-k) tinv(0.975,T-k) tinv(0.995,T-k)];
pval = 2 * (1-tcdf(tstat,T-k));
disp('question 9')
disp(' ')
disp("beta estimates:")
disp(["constant:   " + num2str(beta(1))])
disp(["inf:        " + num2str(beta(2))])
disp(' ')
disp("coefficient standard deviations:")
disp(["constant:   " + num2str(stderr(1))])
disp(["inflation:  " + num2str(stderr(2))])
disp(' ')
disp("t-stats for the two-sided t-test:")
disp(["constant:   " + num2str(tstat(1))])
disp(["inflation:  " + num2str(tstat(2))])
disp("critical values at 10%, 5% and 1% are respectively " + num2str(critvals(1)) + ", " + num2str(critvals(2)) + ", and " + num2str(critvals(3)))
disp(' ')
disp("p-values for the two-sided t-test:")
disp(["constant:   " + num2str(pval(1))])
disp(["inflation:  " + num2str(pval(2))])
disp(' ')
disp(' ')


% question 10

fig_10_1 = figure;
set(fig_10_1, 'color', [.92 .92 .90], "units", "pixels", "position", [300, 300, 600, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 10: plot of Phillips curve residuals")
plot(time, [nan;u], 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
xlabel('time periods')
ylabel('residuals')
xlim([1960 2012.75])
grid on
title('question 10, figure 6: plot of Phillips curve residuals', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'normal');
print('-clipboard', '-dbitmap')

u_t = u(2:end);
u_t_1 = u(1:end-1);
rho = (u_t_1'*u_t_1)\(u_t_1'*u_t);
e = u_t - u_t_1*rho;
s2 = e'*e/(numel(u_t)-1);
S = 1/(u_t_1'*u_t_1);
stderr = diag(s2*S).^0.5;
tstat = abs(rho./stderr);
critvals = [tinv(0.95,numel(u_t)-1) tinv(0.975,numel(u_t)-1) tinv(0.995,numel(u_t)-1)];
pval = 2 * (1-tcdf(tstat,numel(u_t)-1));
disp('question 10')
disp(' ')
disp("rho estimate:                        " + num2str(rho))
disp("coefficient standard deviation:      " + num2str(stderr))
disp("t-stat for the two-sided t-test:     " + num2str(tstat))
disp("critical values at 10%, 5% and 1%:   " + num2str(critvals(1)) + ", " + num2str(critvals(2)) + ", and " + num2str(critvals(3)))
disp("p-value for the two-sided t-test:    " + num2str(pval))
disp(' ')
DW = sum((u_t - u_t_1).^2) / (u_t'*u_t);
disp("DW stat:         " + num2str(DW))
disp("dL:              " + num2str(1.653))
disp("dU:              " + num2str(1.693))
disp(' ')
disp("please refer to figure 6")
disp(' ')
disp(' ')


% question 11

y_tld = [sqrt(1-rho^2)*unemp(1) ;unemp(2:end)-rho*unemp(1:end-1)];
X_tld = [sqrt(1-rho^2) sqrt(1-rho^2)*inf(1);repmat(1-rho, [numel(inf)-1,1]) inf(2:end)-rho*inf(1:end-1)];
[T,k] = size(X_tld);
beta = (X_tld'*X_tld)\(X_tld'*y_tld);
u = y_tld - X_tld*beta;
s2 = u'*u/(T-k);
S = (X_tld'*X_tld)\eye(k);
stderr = diag(s2*S).^0.5;
tstat = abs(beta./stderr);
critvals = [tinv(0.95,T-k) tinv(0.975,T-k) tinv(0.995,T-k)];
pval = 2 * (1-tcdf(tstat,T-k));
fig_11_1 = figure;
set(fig_11_1, 'color', [.92 .92 .90], "units", "pixels", "position", [300, 300, 600, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 11: plot of Phillips curve residuals (FGLS)")
plot(time, [nan;u], 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
xlabel('time periods')
ylabel('residuals')
xlim([1960 2012.75])
grid on
title('question 11, figure 7: plot of Phillips curve residuals (FGLS)', 'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'normal');
print('-clipboard', '-dbitmap')
disp('question 11')
disp(' ')
disp("beta estimates (FGLS):")
disp(["constant:   " + num2str(beta(1))])
disp(["inf:        " + num2str(beta(2))])
disp(' ')
disp("coefficient standard deviations:")
disp(["constant:   " + num2str(stderr(1))])
disp(["inflation:  " + num2str(stderr(2))])
disp(' ')
disp("t-stats for the two-sided t-test:")
disp(["constant:   " + num2str(tstat(1))])
disp(["inflation:  " + num2str(tstat(2))])
disp("critical values at 10%, 5% and 1% are respectively " + num2str(critvals(1)) + ", " + num2str(critvals(2)) + ", and " + num2str(critvals(3)))
disp(' ')
disp("p-values for the two-sided t-test:")
disp(["constant:   " + num2str(pval(1))])
disp(["inflation:  " + num2str(pval(2))])
disp(' ')
disp("please refer to figure 7")
disp(' ')
disp(' ')


% question 12

dummy = [zeros(105,1); ones(106,1)];
y = y_tld;
X = [X_tld dummy dummy.*inf];
[n, k] = size(X);
beta = (X'*X)\(X'*y);
u = y - X*beta;
S = (X'*X)\eye(k);
R = [0 0 1 0; 0 0 0 1];
r = [0; 0];
q = numel(r);
F = ( (R*beta-r)' * inv(R*S*R') * (R*beta-r) / q) / ( u'*u / (n-k) );
critval = finv(0.95,q,n-k);
pval = 1 - fcdf(F,q,n-k);
disp('question 12')
disp(' ')
disp("F-stat for the joint hypothesis:    " + num2str(F))
disp("critical value at 5%:               " + num2str(critval))
disp("p-value for the joint hypothesis:   " + num2str(pval))
disp(' ')
disp(' ')


% question 13

n = numel(unemp);
tau0 = round(0.15 * n);
tau1 = round(0.85 * n);
y = y_tld;
Fmax= -1000;
for tau = tau0:tau1
    dummy = [zeros(tau,1); ones(n-tau,1)];
    X = [X_tld dummy dummy.*inf];
    [n, k] = size(X);
    beta = (X'*X)\(X'*y);
    u = y - X*beta;
    S = (X'*X)\eye(k);
    R = [0 0 1 0; 0 0 0 1];
    r = [0; 0];
    q = numel(r);
    F = ( (R*beta-r)' * inv(R*S*R') * (R*beta-r) / q) / ( u'*u / (n-k) );
    if F > Fmax
        Fmax = F;
        break_tau = tau;
    end
end
critval = finv(0.95,q,n-k);
pval = 1 - fcdf(Fmax,q,n-k);  
break_period = time(break_tau+1);
disp('question 13')
disp(' ')
disp("max F-stat for the joint hypothesis (QLR test):    " + num2str(Fmax))
disp("critical value at 10%:                             " + num2str(5.00))
disp("critical value at  5%:                             " + num2str(5.86))
disp("critical value at  1%:                             " + num2str(7.78))
disp(' ')
disp("The optimal period for the sutructural break is " + num2str(break_period));
disp(' ')
disp(' ')


% question 14

y = unemp(5:end);
X = [ones(numel(inf)-4,1) inf(4:end-1) inf(3:end-2) inf(2:end-3) inf(1:end-4) ...
    unemp(4:end-1) unemp(3:end-2) unemp(2:end-3) unemp(1:end-4)];
[n, k] = size(X);
beta = (X'*X)\(X'*y);
u = y - X*beta;
S = (X'*X)\eye(k);
R = [zeros(4,1) eye(4) zeros(4,4)];
r = zeros(4,1);
q = numel(r);
F = ( (R*beta-r)' * inv(R*S*R') * (R*beta-r) / q) / ( u'*u / (n-k) );
critval = finv(0.95,q,n-k);
pval = 1 - fcdf(F,q,n-k);
disp('question 14')
disp(' ')
disp("F-stat for the joint hypothesis (no Granger causality):    " + num2str(F))
disp("critical value at 5%:                                      " + num2str(critval))
disp("p-value for the joint hypothesis:                          " + num2str(pval))
disp(' ')
disp(' ')


% question 15

dl_inf = [0;beta(2:5)];
dl_unemp = [0;beta(6:9)];
period = (0:4)';
fig_15_1 = figure;
set(fig_15_1, 'color', [.92 .92 .90], "units", "pixels", "position", [300, 300, 1200, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 15: distributed lags")
subplot(1, 2, 1)
plot(period, dl_inf, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
xlabel('periods')
ylabel('dl for inflation')
xlim([0 4])
subplot(1, 2, 2)
plot(period, dl_unemp, 'LineWidth', 2, 'Color', [0.2 0.3 0.8])
xlabel('periods')
ylabel('dl for unemployment')
xlim([0 4])
sgtitle('question 15, figure 8: distributed lags for inflation and unemployment')
print('-clipboard', '-dbitmap')
lr_impact = sum(dl_inf);
disp('question 15')
disp(' ')
disp("The long-run impact of inflation on unemployment is " + num2str(lr_impact))
disp(' ')
disp(' ')




