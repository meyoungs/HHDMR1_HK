close all
clear
clc
addpath('Functions\')

%% Problem settings
% rng('shuffle')
rng(1) % for reproduction

N = 100; % # of test for constructed metamodel
FE_th = 1999;
n_MC = 1e5;
c = 3;

mu = [1.5, 2.5];
sigma_ = ones(1, 2);
dim = size(mu, 2);

func = @prob_2D;
lb = -c*sigma_.*ones(1, dim) + mu;
ub = c*sigma_.*ones(1, dim) + mu;
g_th = 2;

%% generate MCS points
MCS_x = normrnd(0, 1, [n_MC, dim]).*sigma_ + mu;

%% True Pf
true_y = func(MCS_x);
true_Pf = nnz(true_y>g_th)/n_MC;
disp(['True probability of failure: ', num2str(true_Pf)])

%% Initial DoE
X = mu.*ones(2*dim+1, dim);
for i = 1:dim
    X(2*i, i) = ub(i);
    X(2*i+1, i) = lb(i);
end
Y = func(X);
N0 = size(X, 1); % F.E for initial sampling
x0 = X(1,:);
y0 = Y(1); % output value at mean

%% Settings for Step1
theta0_1D = ones(1, 1);
lb_theta_1D = 1e-5*ones(1, 1);
ub_theta_1D = 1e2*ones(1, 1);

X_cell = cell(dim, 1);
Y_cell = cell(dim, 1);

for i = 1:dim
    X_cell{i} = [x0(i); X(2*i:2*i+1, i)];
    Y_cell{i} = [y0; Y(2*i:2*i+1)];
end

FE = size(X, 1); % # of the function evaluation

%% Step 1: AK-HDMR1
while true
    while true
        % build metamodel
        dmodel_cell = cell(dim, 1);
        for idx = 1:dim
            [dmodel, ~] = dacefit(X_cell{idx}, Y_cell{idx}, @regpoly0, @corrgauss, theta0_1D, lb_theta_1D, ub_theta_1D);
            dmodel_cell{idx} = dmodel;
        end

        % prediction of metamodel
        [pred_mu, pred_var, pred_var_array] = HDMR1_pred(MCS_x, y0, dmodel_cell);

        Nf = nnz(pred_mu > g_th);
        Pf = Nf/n_MC;

        % MCE learning function
        CL = normcdf(abs(pred_mu-g_th)./sqrt(pred_var));
        CCL = sum(CL)/n_MC;
        if CCL > 0.9999
            break
        end
        
        pdf_vec = zeros(size(MCS_x, 1), dim);
        for i = 1:dim
            pdf_vec(:, i) = normpdf((MCS_x(:,i)-mu(i))./sigma_(i), 0, 1);
        end
        pdf_vec = prod(pdf_vec, 2);
        
        EI = (1-CL).*pdf_vec.*sqrt(pred_var);
        [~, iidx]=max(EI);

        % identify dimension that has the largest variance
        [~, dimidx] = max(pred_var_array(iidx, :));

        % subspace sampling
        newX = MCS_x(iidx, :); % new_norm_x: 1*dim
        proj_newX = newX(dimidx);
        [proj_X, full_proj_X] = subsamp(proj_newX, X_cell, dmodel_cell, dimidx, mu);

        % check multiple design point
        if ~min(sum(abs(X - full_proj_X), 2))
            disp('multiple design point')
            MCS_x(iidx, :) = normrnd(0, 1, [1, dim]).*sigma_ + mu;
            true_y(iidx) = func(MCS_x(iidx,:));
            continue
        end
        X = [X; full_proj_X];
        new_Y = func(full_proj_X);
        Y = [Y; new_Y];
        FE = size(Y, 1);

        % update training set
        X_cell{dimidx} = [X_cell{dimidx}; proj_X];
        Y_cell{dimidx} = [Y_cell{dimidx}; new_Y];

        save('HHDMR1_HK_step1_result.mat', "true_Pf", "FE", "Pf", "N0", "n_MC")

    end
    save('HHDMR1_HK_step1_result.mat', "true_Pf", "FE", "Pf", "N0", "n_MC")
    COV = sqrt((1-Pf)/(Pf*n_MC));

    if COV < 0.1
        disp(['Estimated probability of failure: ', num2str(Pf)])
        disp(['COV: ', num2str(COV)])
        disp(['The number of functions evaluations: ', num2str(FE)])
        break
    else
        disp('more MCS points are needed');
        MCS_x = [MCS_x; normrnd(0, 1, [n_MC, dim]).*sigma_ + mu];
        n_MC = size(MCS_x, 1);
        true_y = func(MCS_x);
    end
end

%% Step1 results
HDMR1_pred_mu = HDMR1_pred(MCS_x, y0, dmodel_cell);
FHDMR1_pred_mu = FHDMR1_pred(MCS_x, y0, dmodel_cell);

Pf_HDMR1 = nnz(HDMR1_pred_mu > g_th) / n_MC;
Pf_FHDMR1 = nnz(FHDMR1_pred_mu > g_th) / n_MC;

N1 = size(Y, 1);
disp('end of Step1')

%% Settings for Step2
gamma_par = struct('g_th', g_th, 'y0', y0);
stop_crit_lst = [];
gamma_lst = [];
theta0 = ones(1, dim);
lb_theta = 1e-5*ones(1, dim);
ub_theta = 1e3*ones(1, dim);

%% Step2: AK-HK
while true
    while true
        if (size(Y, 1)>FE_th)
            gamma_ = gamma_est_weighted(X(N1+1:end, :), Y(N1+1:end), gamma_par, dmodel_cell);
            break
        end

        if size(Y, 1) > N1
            gamma_ = gamma_est_weighted(X(N1+1:end, :), Y(N1+1:end), gamma_par, dmodel_cell);
        else
            gamma_ = 1;
        end
        gamma_lst = [gamma_lst, gamma_];
        HK_model = HK_fit(X, Y, @(x) HHDMR1_pred(x, y0, dmodel_cell, gamma_), lb_theta, ub_theta);

        % prediction of HK
        [pred_mu, pred_var] = HK_pred(MCS_x, HK_model);
        HHDMR1_mu = HHDMR1_pred(MCS_x, y0, dmodel_cell, gamma_);

        Pf = nnz(pred_mu>g_th)/n_MC;
        HHDMR1_Pf = nnz(HHDMR1_mu>g_th)/n_MC;

        % MCE learning function
        CL = normcdf(abs(pred_mu-g_th)./sqrt(pred_var));
        CCL = sum(CL)/n_MC;
        stop_crit_lst = [stop_crit_lst, CCL];
        if (size(Y, 1) > (N1+5)) && (CCL > 0.9999)
            gamma_ = gamma_est_weighted(X(N1+1:end, :), Y(N1+1:end), gamma_par, dmodel_cell);
            break
        end
        
        pdf_vec = zeros(size(MCS_x, 1), dim);
        for i = 1:dim
            pdf_vec(:, i) = normpdf((MCS_x(:,i)-mu(i))./sigma_(i), 0, 1);
        end
        pdf_vec = prod(pdf_vec, 2);

        if size(Y, 1) == N1
            EI = pdf_vec.*exp(-((pred_mu-g_th)/(g_th)).^2);
            [~, idx]=max(EI);
        else
            EI = (1-CL).*pdf_vec.*sqrt(pred_var);
            [~, idx]=max(EI);
        end

        newX = MCS_x(idx, :);

        if ~min(sum(abs(X-newX), 2))
            disp('multiple design point')
            MCS_x(idx, :) = normrnd(0, 1, [1, dim]).*sigma_ + mu;
            true_y(idx) = func(MCS_x(idx, :));
            continue
        end

        % update dataset
        X = [X; newX];
        new_Y = func(newX);
        Y = [Y; new_Y];
        FE = size(Y, 1);
        
        save('HHDMR1_HK_step2_result.mat', "true_Pf", "FE", "Pf", "n_MC", "HHDMR1_Pf", "CCL", "N1", "gamma_", "gamma_lst", "stop_crit_lst")
    end
    save('HHDMR1_HK_step2_result.mat', "true_Pf", "FE", "Pf", "n_MC", "HHDMR1_Pf", "CCL", "N1", "gamma_", "gamma_lst", "stop_crit_lst")
    COV = sqrt((1-Pf)/(Pf*n_MC));

    if COV < 0.05
        disp(['Estimated probability of failure: ', num2str(Pf)])
        disp(['COV: ', num2str(COV)])
        disp(['The number of functions evaluations: ', num2str(FE)])
        break
    else
        disp('more MCS points are needed');
        MCS_x = [MCS_x; normrnd(0, 1, [n_MC, dim]).*sigma_ + mu];
        n_MC = size(MCS_x, 1);
        true_y = func(MCS_x);
    end
end
disp('end of Step2')

%% Step2 results
HK_model = HK_fit(X, Y, @(x) HHDMR1_pred(x, y0, dmodel_cell, gamma_), lb_theta, ub_theta);
FE = size(Y, 1);
save('HHDMR1_HK_step2_result.mat', "true_Pf", "FE", "Pf", "n_MC", "HHDMR1_Pf", "CCL", "N1", "gamma_", "gamma_lst", "stop_crit_lst")

%% check metamodel
true_Pf_lst = zeros(1, N);
HHDMR1_HK_Pf_lst = zeros(1, N);
HHDMR1_Pf_lst = zeros(1, N);

for kkk = 1:N
    MCS_x = normrnd(0, 1, [n_MC, dim]).*sigma_ + mu;
    
    HHDMR1_Pf = nnz(HHDMR1_pred(MCS_x, y0, dmodel_cell, gamma_) > g_th) / n_MC;
    HHDMR1_Pf_lst(kkk) = HHDMR1_Pf;
    
    pred_mu = HK_pred(MCS_x, HK_model);
    HHDMR1_HK_Pf = nnz(pred_mu > g_th) / n_MC;
    HHDMR1_HK_Pf_lst(kkk) = HHDMR1_HK_Pf;

    true_y = func(MCS_x);
    true_Pf = nnz(true_y > g_th) / n_MC;
    true_Pf_lst(kkk) = true_Pf;
end
rowNames = {'Pf'};  % Row names
data = table([mean(true_Pf_lst)], [mean(HHDMR1_HK_Pf_lst)], [mean(HHDMR1_Pf_lst)], ...
    'VariableNames', {'Pf (MCS)', 'Pf (proposed method)', 'Pf (HHDMR1 method)'}, ...
    'RowNames', rowNames);
disp(data)

name = strcat('HHDMR1_HK_result');
save(name)

%% plot
initX = X(1:N0, :);
step1X = X(N0+1:N1, :);
step2X = X(N1+1:end, :);

grid_n = 80;
new_X = gridsamp([-5 -5; 5 5], grid_n);
new_X = sigma_.*new_X+mu;
minX = min(new_X);
maxX = max(new_X);

true_Y = func(new_X);
[new_HHDMR1_HK_pred, new_HHDMR1_HK_pred_var] = HK_pred(new_X, HK_model);
new_HDMR1_pred = HDMR1_pred(new_X, y0, dmodel_cell);
new_FHDMR1_pred = FHDMR1_pred(new_X, y0, dmodel_cell);
new_HHDMR1_pred = HHDMR1_pred(new_X, y0, dmodel_cell, gamma_);

[dmodel, ~] = dacefit(X, Y, @regpoly0, @corrgauss, ones(1,dim), 0.0001*ones(1,dim), 100*ones(1, dim));
[pred_mu_dace, pred_var_dace] = predictor(new_X, dmodel);
X1 = reshape(new_X(:, 1), grid_n, grid_n); X2 = reshape(new_X(:, 2), grid_n, grid_n);
true_Y = reshape(true_Y, grid_n, grid_n);
new_HHDMR1_HK_pred = reshape(new_HHDMR1_HK_pred, grid_n, grid_n);
new_HHDMR1_HK_pred_var = reshape(new_HHDMR1_HK_pred_var, grid_n, grid_n);
new_HDMR1_pred = reshape(new_HDMR1_pred, grid_n, grid_n);
new_FHDMR1_pred = reshape(new_FHDMR1_pred, grid_n, grid_n);
new_HHDMR1_pred = reshape(new_HHDMR1_pred, grid_n, grid_n);
pred_mu_dace = reshape(pred_mu_dace, grid_n, grid_n);
pred_var_dace = reshape(pred_var_dace, grid_n, grid_n);
cv = [g_th, g_th+0.001];

figure()
hold on
axis([minX(1) maxX(1), minX(2), maxX(2)])
s1 = plot(X(:,1), X(:,2), 'r*', 'MarkerSize', 7, 'MarkerFaceColor', 'r');
contour(X1, X2, true_Y, cv,'k-', 'LineWidth', 2);
contour(X1, X2, new_HDMR1_pred, cv, 'c-.', 'LineWidth', 2);
contour(X1, X2, new_FHDMR1_pred, cv, 'g-.', 'LineWidth', 2);
contour(X1, X2, new_HHDMR1_pred, cv, 'r-', 'LineWidth', 2);
contour(X1, X2, new_HHDMR1_HK_pred, cv, 'b-', 'LineWidth', 2);
l1 = plot(nan, nan, 'k-', 'LineWidth', 2);
l2 = plot(nan, nan, 'c-.', 'LineWidth', 2);
l3 = plot(nan, nan, 'g-.', 'LineWidth', 2);
l4 = plot(nan, nan, 'r-', 'LineWidth', 2);
l5 = plot(nan, nan, 'b-', 'LineWidth', 2);
grid on
legend([s1, l1, l2, l3, l4, l5], {'Samples', 'True Pf', 'Est. Pf(HDMR1)', 'Est. Pf(FHDMR1)', 'Est. Pf(HHDMR1)', 'Est. Pf(Proposed)'}, 'Location', 'best')
hold off