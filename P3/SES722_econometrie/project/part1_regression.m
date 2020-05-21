% preliminaries
clear
clc
close all
% cd 'path_to_data'; % if needed, to change working directory



%--------------------------
% Part 1: regression      |
%--------------------------


% question 1

load ('mroz.txt')
[nobs, nvar] = size(mroz);
data = mroz(mroz(:,7) > 0,:);
[nobs_trim, ~] = size(data);
disp('question 1')
disp(' ')
disp(['number of variables in the mroz dataset: ' num2str(nvar)])
disp(['number of observations in the initial dataset: ' num2str(nobs)])
disp(['number of observations after trimming non-positive wage values: ' num2str(nobs_trim)])
disp(' ')
disp(' ')


% question 2

labels = ["age", "educ", "wage"];
pctile65 = prctile(data(:,12), 65);
data1 = data(:, 5:7);
data2 = data(data(:,12) >= pctile65, 5:7);
data3 = data(data(:,12) < pctile65, 5:7);
statistics1 = [" " labels; "count" string(repmat(nobs_trim, 1, 3)); "mean" string(mean(data1)); "median" string(median(data1)); ...
    "std" string(std(data1)); "min" string(min(data1)); "25%" string(prctile(data1, 25)); "50%" string(prctile(data1, 50));...
    "75%" string(prctile(data1, 75)); "max" string(max(data1))];
statistics2 = [" " labels; "count" string(repmat(nobs_trim, 1, 3)); "mean" string(mean(data2)); "median" string(median(data2)); ...
    "std" string(std(data2)); "min" string(min(data2)); "25%" string(prctile(data2, 25)); "50%" string(prctile(data2, 50));...
    "75%" string(prctile(data2, 75)); "max" string(max(data2))];
statistics3 = [" " labels; "count" string(repmat(nobs_trim, 1, 3)); "mean" string(mean(data3)); "median" string(median(data3)); ...
    "std" string(std(data3)); "min" string(min(data3)); "25%" string(prctile(data3, 25)); "50%" string(prctile(data3, 50));...
    "75%" string(prctile(data3, 75)); "max" string(max(data3))];
disp('question 2')
disp(' ')
disp("descriptive statistics for all women:")
disp(statistics1)
disp("descriptive statistics for women whose husband's income is more than 65% of the sample:")
disp(statistics2)
disp("descriptive statistics for women whose husband's income is less than 65% of the sample:")
disp(statistics3)
disp(' ')


% question 3

wage = data(:, 7);
mean_wage = mean(wage);
std_wage = std(wage);
wage_trimmed = wage(wage > mean_wage-3*std_wage & wage < mean_wage+3*std_wage);
fig_3_1 = figure;
set(fig_3_1, "units", "pixels", "position", [300, 300, 1200, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 3: histograms of wage")
subplot(1, 2, 1)
histogram(wage, 30, 'FaceColor', [0.2,0.2,0.7])
title('question 3, figure 1: histogram of wage', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
subplot(1, 2, 2)
histogram(wage_trimmed, 30, 'FaceColor', [0.2,0.2,0.7])
title('question 3, figure 2: histogram of wage, outliers trimmed', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
print('-clipboard', '-dbitmap')
disp('question 3')
disp(' ')
disp("please refer to figure 1")
disp(' ')
disp(' ')


% question 4

coefficient = [0 1] * corrcoef(data(:,15:16)) * [1;0];
disp('question 4')
disp(' ')
disp(["correlation coefficient between motheduc and fatheduc: " + num2str(coefficient)])
disp(' ')
disp(' ')


% question 5

educ = data(:, 6);
fig_5_1 = figure;
set(fig_5_1, "units", "pixels", "position", [300, 300, 600, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 5: scatterplot of wage and educ")
scatter(wage, educ, 35, 'filled', "MarkerEdgeColor", [0.3 0.3 0.3], "MarkerFaceColor", [1 0.4 0.1], "LineWidth", 0.5);
title('question 5, figure 2: scatterplot of wage and educ', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
print('-clipboard', '-dbitmap')
disp('question 5')
disp(' ')
disp("please refer to figure 2")
disp(' ')
disp(' ')


% question 6

disp('question 6')
disp(' ')
disp("please refer to question 6 in the report")
disp(' ')
disp(' ')


% question 7

y = data(:,21);
X = [ones(nobs_trim,1) data(:,[18, 6, 19, 20, 3, 4])];
[n, k] = size(X);
beta = (X'*X)\(X'*y);
u = y - X*beta;
s2 = u'*u/(n-k);
S = (X'*X)\eye(k);
stderr = diag(s2*S).^0.5;
fig_7_1 = figure;
set(fig_7_1, "units", "pixels", "position", [300, 300, 600, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 7: histograms of residuals")
histogram(u, 30, 'FaceColor', [0.2,0.2,0.7])
title('question 7, figure 3: histogram of residuals u', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
print('-clipboard', '-dbitmap')
disp('question 7')
disp(' ')
disp("beta estimates:")
disp(["constant:   " + num2str(beta(1))])
disp(["city:        " + num2str(beta(2))])
disp(["educ:        " + num2str(beta(3))])
disp(["exper:       " + num2str(beta(4))])
disp(["nwifeinc:    " + num2str(beta(5))])
disp(["kidslt6:    " + num2str(beta(6))])
disp(["kidsgt6:    " + num2str(beta(7))])
disp(' ')
disp("coefficient standard deviations:")
disp(["constant:   " + num2str(stderr(1))])
disp(["city:       " + num2str(stderr(2))])
disp(["educ:       " + num2str(stderr(3))])
disp(["exper:      " + num2str(stderr(4))])
disp(["nwifeinc:   " + num2str(stderr(5))])
disp(["kidslt6:    " + num2str(stderr(6))])
disp(["kidsgt6:    " + num2str(stderr(7))])
disp(' ')
disp("please refer to figure 3")
disp(' ')
disp(' ')


% question 8

tstat = abs(beta./stderr);
critvals = [tinv(0.95,n-k) tinv(0.975,n-k) tinv(0.995,n-k)];
pval = 2 * (1-tcdf(tstat,n-k));
disp('question 8')
disp(' ')
disp("t-stats for the two-sided t-test:")
disp(["constant:   " + num2str(tstat(1))])
disp(["city:       " + num2str(tstat(2))])
disp(["educ:       " + num2str(tstat(3))])
disp(["exper:      " + num2str(tstat(4))])
disp(["nwifeinc:   " + num2str(tstat(5))])
disp(["kidslt6:    " + num2str(tstat(6))])
disp(["kidsgt6:    " + num2str(tstat(7))])
disp("critical values at 10%, 5% and 1% are respectively " + num2str(critvals(1)) + ", " + num2str(critvals(2)) + ", and " + num2str(critvals(3)))
disp(' ')
disp("p-values for the two-sided t-test:")
disp(["constant:   " + num2str(pval(1))])
disp(["city:       " + num2str(pval(2))])
disp(["educ:       " + num2str(pval(3))])
disp(["exper:      " + num2str(pval(4))])
disp(["nwifeinc:   " + num2str(pval(5))])
disp(["kidslt6:    " + num2str(pval(6))])
disp(["kidsgt6:    " + num2str(pval(7))])
disp(' ')
disp(' ')


% question 9

tstat_nwifeinc = abs((beta(5)-0.01)./stderr(5));
pval_nwifeinc = 2 * (1-tcdf(tstat_nwifeinc,n-k));
disp('question 9')
disp(' ')
disp("t-stat for the two-sided t-test:    " + num2str(tstat_nwifeinc))
disp("p-value for the two-sided t-test:   " + num2str(pval_nwifeinc))
disp(' ')
disp(' ')


% question 10

R = [0 0 0 0 1 0 0; 0 1 0 0 0 0 0];
r = [0.01; 0.05];
q = numel(r);
F = ( (R*beta-r)' * inv(R*S*R') * (R*beta-r) / q) / ( u'*u / (n-k) );
critval = finv(0.95,q,n-k);
pval = 1 - fcdf(F,q,n-k);
disp('question 10')
disp(' ')
disp("F-stat for the joint hypothesis:    " + num2str(F))
disp("critical value at 5%:               " + num2str(critval))
disp("p-value for the joint hypothesis:   " + num2str(pval))
disp(' ')
disp(' ')


% question 11

R = [0 1 0 0 1 0 0; 0 0 1 1 0 0 0];
r = [0.1; 0.1];
q = numel(r);
F = ( (R*beta-r)' * inv(R*S*R') * (R*beta-r) / q) / ( u'*u / (n-k) );
critval = finv(0.95,q,n-k);
pval = 1 - fcdf(F,q,n-k);
disp('question 11')
disp(' ')
disp("F-stat for the joint hypothesis:    " + num2str(F))
disp("critical value at 5%:               " + num2str(critval))
disp("p-value for the joint hypothesis:   " + num2str(pval))
disp(' ')
disp(' ')


% question 12

[ax_educ,ax_exper] = meshgrid(5:20,0:40);
ax_wage = exp(beta(1) + beta(3)*ax_educ + beta(4)*ax_exper);
fig_12_1 = figure;
set(fig_12_1, "units", "pixels", "position", [300, 300, 600, 450], "MenuBar", "none", "NumberTitle", "off", "name", "question 12: Effect of educ and exper on wage")
surf(ax_educ, ax_exper, ax_wage)
xlabel('educ')
ylabel('exper')
zlabel('wage')
title('question 12, figure 4: effect of educ and exper on wage', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
print('-clipboard', '-dbitmap')
disp('question 12')
disp(' ')
disp("please refer to figure 4")
disp(' ')
disp(' ')


% question 13

R = [0 0 0 0 0 1 -1];
r = [0];
q = numel(r);
F = ( (R*beta-r)' * inv(R*S*R') * (R*beta-r) / q) / ( u'*u / (n-k) );
critval = finv(0.95,q,n-k);
pval = 1 - fcdf(F,q,n-k);
disp('question 13')
disp(' ')
disp("F-stat for the joint hypothesis:    " + num2str(F))
disp("critical value at 5%:               " + num2str(critval))
disp("p-value for the joint hypothesis:   " + num2str(pval))
disp(' ')
disp(' ')


% question 14

% test for linear heteroskedasticity on initial model
u2 = u.^2;
delta = (X'*X)\(X'*u2);
v = u2 - X*delta;
R = [zeros(6,1) eye(6)];
r = zeros(6,1);
q = numel(r);
F = ( (R*delta-r)' * inv(R*S*R') * (R*delta-r) / q) / ( v'*v / (n-k) );
critval = finv(0.95,q,n-k);
pval = 1 - fcdf(F,q,n-k);
disp('question 14')
disp(' ')
disp("F-stat for linear heteroskedasticity:    " + num2str(F))
disp("critical value at 5%:                    " + num2str(critval))
disp("p-value for the joint hypothesis:        " + num2str(pval))
disp(' ')
% scatterplots of relations between log(wage) and other variables
labels = ['const', 'city', "educ", 'exper', 'nwifeinc', 'kidslt6', 'kidsge6'];
fig_14_1 = figure;
set(fig_14_1, "units", "pixels", "position", [300, 300, 1200, 500], "MenuBar", "none", "NumberTitle", "off", "name", "question 12: Effect of educ and exper on wage")
sgtitle('question 14, figure 5: scatterplots of log(wage) with other variables', 'FontName', 'Times New Roman', 'FontSize', 10, 'FontWeight', 'normal');
for index=1:numel(labels)
    subplot(2, 4, index)
    scatter(y, X(:,index), 35, 'filled', "MarkerEdgeColor", [0.3 0.3 0.3], "MarkerFaceColor", [0.1 1 0.4], "LineWidth", 0.5);
    title(['log(wage)-'+labels(index)], 'FontName', 'Times New Roman', 'FontSize', 8, 'FontWeight', 'normal');
end
print('-clipboard', '-dbitmap')
% creation of transformed variables and estimation of transformed model
X_2 = [X(:,1) log(X(:,3)) log(X(:,4)+1) log(X(:,5)+1+abs(min(X(:,5)))) X(:,2) X(:,6)==0 X(:,6)==1 X(:,7)==0 X(:,7)==1 X(:,7)==2 X(:,7)==3 X(:,7)==4];
[n, k_2] = size(X_2);
beta_2=(X_2'*X_2)\(X_2'*y);
u_2 = y - X_2*beta_2;
s2_2 = u_2'*u_2/(n-k_2);
S_2 = (X_2'*X_2)\eye(k_2);
stderr_2 = diag(s2_2*S_2).^0.5;
disp("beta estimates for the transformed model:")
disp(["constant:         " + num2str(beta_2(1))])
disp(["log(educ):         " + num2str(beta_2(2))])
disp(["log(exper):        " + num2str(beta_2(3))])
disp(["log(nwifeinc):     " + num2str(beta_2(4))])
disp(["city:              " + num2str(beta_2(5))])
disp(["kidslt6_0:         " + num2str(beta_2(6))])
disp(["kidslt6_1:         " + num2str(beta_2(7))])
disp(["kidsgt6_0:        " + num2str(beta_2(8))])
disp(["kidsgt6_1:        " + num2str(beta_2(9))])
disp(["kidsgt6_2:        " + num2str(beta_2(10))])
disp(["kidsgt6_3:        " + num2str(beta_2(11))])
disp(["kidsgt6_4:         " + num2str(beta_2(12))])
disp(' ')
disp("coefficient standard deviations:")
disp(["constant:         " + num2str(stderr_2(1))])
disp(["log(educ):        " + num2str(stderr_2(2))])
disp(["log(exper):       " + num2str(stderr_2(3))])
disp(["log(nwifeinc):    " + num2str(stderr_2(4))])
disp(["city:             " + num2str(stderr_2(5))])
disp(["kidslt6_0:        " + num2str(stderr_2(6))])
disp(["kidslt6_1:        " + num2str(stderr_2(7))])
disp(["kidsgt6_0:        " + num2str(stderr_2(8))])
disp(["kidsgt6_1:        " + num2str(stderr_2(9))])
disp(["kidsgt6_2:        " + num2str(stderr_2(10))])
disp(["kidsgt6_3:        " + num2str(stderr_2(11))])
disp(["kidsgt6_4:        " + num2str(stderr_2(12))])
disp(' ')
u2_2 = u_2.^2;
delta_2 = (X_2'*X_2)\(X_2'*u2_2);
v_2 = u2_2 - X_2*delta_2;
R = [zeros(11,1) eye(11)];
r = zeros(11,1);
q = numel(r);
F = ( (R*delta_2-r)' * inv(R*S_2*R') * (R*delta_2-r) / q) / ( v_2'*v_2 / (n-k_2) );
critval = finv(0.95,q,n-k_2);
pval = 1 - fcdf(F,q,n-k_2);
disp("F-stat for linear heteroskedasticity (tranformed model):    " + num2str(F))
disp("critical value at 5%:                                       " + num2str(critval))
disp("p-value for the joint hypothesis:                           " + num2str(pval))
disp(' ')
% estimation of GLS model
logu2 = log(u2);
delta_3 = (X'*X)\(X'*logu2);
g = X*delta_3;
h = exp(g);
y_3 = y./sqrt(h);
X_3 = X./repmat(sqrt(h),[1,k]);
beta_3 = (X_3'*X_3)\(X_3'*y_3);
u_3 = y_3 - X_3*beta_3;
s2_3 = u_3'*u_3/(n-k);
S_3 = (X_3'*X_3)\eye(k);
stderr_3 = diag(s2_3*S_3).^0.5;
disp("beta estimates with GLS:")
disp(["constant:   " + num2str(beta_3(1))])
disp(["city:        " + num2str(beta_3(2))])
disp(["educ:        " + num2str(beta_3(3))])
disp(["exper:       " + num2str(beta_3(4))])
disp(["nwifeinc:    " + num2str(beta_3(5))])
disp(["kidslt6:    " + num2str(beta_3(6))])
disp(["kidsgt6:    " + num2str(beta_3(7))])
disp(' ')
disp("coefficient standard deviations:")
disp(["constant:   " + num2str(stderr_3(1))])
disp(["city:       " + num2str(stderr_3(2))])
disp(["educ:       " + num2str(stderr_3(3))])
disp(["exper:      " + num2str(stderr_3(4))])
disp(["nwifeinc:   " + num2str(stderr_3(5))])
disp(["kidslt6:    " + num2str(stderr_3(6))])
disp(["kidsgt6:    " + num2str(stderr_3(7))])
disp(' ')
disp(' ')


% question 15

dummy = data(:,5)>43;
X_4 = [X X.*repmat(dummy,1,k)];
[n, k_4] = size(X_4);
beta_4 = (X_4'*X_4)\(X_4'*y);
u_4 = y - X_4*beta_4;
s2_4 = u_4'*u_4/(n-k_4);
S_4 = (X_4'*X_4)\eye(k_4);
stderr_4 = diag(s2_4*S_4).^0.5;
tstat_4 = abs(beta_4./stderr_4);
pval_d = 2 * (1-tcdf(tstat_4,n-k_4));
disp('question 15')
disp(' ')
disp("beta estimates with structural change:")
disp(["constant:      " + num2str(beta_4(1))])
disp(["city:           " + num2str(beta_4(2))])
disp(["educ:           " + num2str(beta_4(3))])
disp(["exper:          " + num2str(beta_4(4))])
disp(["nwifeinc:       " + num2str(beta_4(5))])
disp(["kidslt6:       " + num2str(beta_4(6))])
disp(["kidsgt6:       " + num2str(beta_4(7))])
disp(["constant_d:     " + num2str(beta_4(8))])
disp(["city_d:        " + num2str(beta_4(9))])
disp(["educ_d:        " + num2str(beta_4(10))])
disp(["exper_d:       " + num2str(beta_4(11))])
disp(["nwifeinc_d:     " + num2str(beta_4(12))])
disp(["kidslt6_d:      " + num2str(beta_4(13))])
disp(["kidsgt6_d:     " + num2str(beta_4(14))])
disp(' ')
disp("coefficient p-values:")
disp(["constant:     " + num2str(pval_d(1))])
disp(["city:         " + num2str(pval_d(2))])
disp(["educ:         " + num2str(pval_d(3))])
disp(["exper:        " + num2str(pval_d(4))])
disp(["nwifeinc:     " + num2str(pval_d(5))])
disp(["kidslt6:      " + num2str(pval_d(6))])
disp(["kidsgt6:      " + num2str(pval_d(7))])
disp(["constant_d:   " + num2str(pval_d(8))])
disp(["city_d:       " + num2str(pval_d(9))])
disp(["educ_d:       " + num2str(pval_d(10))])
disp(["exper_d:      " + num2str(pval_d(11))])
disp(["nwifeinc_d:   " + num2str(pval_d(12))])
disp(["kidslt6_d:    " + num2str(pval_d(13))])
disp(["kidsgt6_d:    " + num2str(pval_d(14))])
disp(' ')
R = [zeros(k) eye(k)];
r = zeros(k,1);
q = numel(r);
F = ( (R*beta_4-r)' * inv(R*S_4*R') * (R*beta_4-r) / q) / ( u_4'*u_4 / (n-k_4) );
critval = finv(0.95,q,n-k_4);
pval = 1 - fcdf(F,q,n-k_4);
disp("F-stat for structural change:         " + num2str(F))
disp("critical value at 5%:                 " + num2str(critval))
disp("p-value for the joint hypothesis:     " + num2str(pval))
disp(' ')
disp(' ')


% question 16

huseduc = data(:,11);
disp('question 16')
disp(' ')
X_5 = [X huseduc];
[n, k_5] = size(X_5);
beta_5 = (X_5'*X_5)\(X_5'*y);
u_5 = y - X_5*beta_5;
s2_5 = u_5'*u_5/(n-k_5);
S_5 = (X_5'*X_5)\eye(k_5);
stderr_5 = diag(s2_5*S_5).^0.5;
tstat_5 = abs(beta_5./stderr_5);
pval_5 = 2 * (1-tcdf(tstat_5,n-k_5));
disp("beta estimates with huseduc:")
disp(["constant:      " + num2str(beta_5(1))])
disp(["city:           " + num2str(beta_5(2))])
disp(["educ:           " + num2str(beta_5(3))])
disp(["exper:          " + num2str(beta_5(4))])
disp(["nwifeinc:       " + num2str(beta_5(5))])
disp(["kidslt6:       " + num2str(beta_5(6))])
disp(["kidsgt6:       " + num2str(beta_5(7))])
disp(["huseduc:       " + num2str(beta_5(8))])
disp(' ')
disp("coefficient p-values:")
disp(["constant:     " + num2str(pval_5(1))])
disp(["city:         " + num2str(pval_5(2))])
disp(["educ:         " + num2str(pval_5(3))])
disp(["exper:        " + num2str(pval_5(4))])
disp(["nwifeinc:     " + num2str(pval_5(5))])
disp(["kidslt6:      " + num2str(pval_5(6))])
disp(["kidsgt6:      " + num2str(pval_5(7))])
disp(["huseduc:      " + num2str(pval_5(8))])
disp(' ')
disp("min of huseduc: " + num2str(min(huseduc)))
disp("max of huseduc: " + num2str(max(huseduc)))
disp(' ')
X_6 = [X (huseduc <= 8) (huseduc > 8 & huseduc <= 12) (huseduc > 12 & huseduc <= 15)];
[n, k_6] = size(X_6);
beta_6 = (X_6'*X_6)\(X_6'*y);
u_6 = y - X_6*beta_6;
s2_6 = u_6'*u_6/(n-k_6);
S_6 = (X_6'*X_6)\eye(k_6);
stderr_6 = diag(s2_6*S_6).^0.5;
tstat_6 = abs(beta_6./stderr_6);
pval_6 = 2 * (1-tcdf(tstat_6,n-k_6));
disp("beta estimates with huseduc as binary variables:")
disp(["constant:      " + num2str(beta_6(1))])
disp(["city:           " + num2str(beta_6(2))])
disp(["educ:           " + num2str(beta_6(3))])
disp(["exper:          " + num2str(beta_6(4))])
disp(["nwifeinc:       " + num2str(beta_6(5))])
disp(["kidslt6:       " + num2str(beta_6(6))])
disp(["kidsgt6:       " + num2str(beta_6(7))])
disp(["huseduc_1:      " + num2str(beta_6(8))])
disp(["huseduc_2:      " + num2str(beta_6(9))])
disp(["huseduc_3:      " + num2str(beta_6(10))])
disp(' ')
disp("coefficient p-values:")
disp(["constant:     " + num2str(pval_6(1))])
disp(["city:         " + num2str(pval_6(2))])
disp(["educ:         " + num2str(pval_6(3))])
disp(["exper:        " + num2str(pval_6(4))])
disp(["nwifeinc:     " + num2str(pval_6(5))])
disp(["kidslt6:      " + num2str(pval_6(6))])
disp(["kidsgt6:      " + num2str(pval_6(7))])
disp(["huseduc_1:    " + num2str(pval_6(8))])
disp(["huseduc_2:    " + num2str(pval_6(9))])
disp(["huseduc_3:    " + num2str(pval_6(10))])
disp(' ')
R = [zeros(3,7) eye(3)];
r = zeros(3,1);
q = numel(r);
F = ( (R*beta_6-r)' * inv(R*S_6*R') * (R*beta_6-r) / q) / ( u_6'*u_6 / (n-k_6) );
critval = finv(0.95,q,n-k_6);
pval = 1 - fcdf(F,q,n-k_6);
disp("F-stat for structural change:         " + num2str(F))
disp("critical value at 95%:                " + num2str(critval))
disp("p-value for the joint hypothesis:     " + num2str(pval))
disp(' ')
disp(' ')





