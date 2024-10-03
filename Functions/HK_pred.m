function [pred_mean, pred_mse] = HK_pred(Xnew, model_info)
    % INPUT
    % newX: unobserved point
    % model_info: information for the HK prediction
    
    % OUTPUT
    % pred_mean: prediction mean at newX
    % pred_mse: prediction variance at newX
    
    % Settings
    minY = model_info.minY;
    maxY = model_info.maxY;
    
    new_ns = size(Xnew, 1);
    
    % Cross correlation vector / squared exponential kernel
    % To avoid errors related to the maximum array size limit, if the number of prediction points exceeds part_num, divide them into part_num parts and compute separately.
    part_num = 1000;
    if new_ns < part_num
        [pred_mean, pred_mse] = calcul_mean_var(Xnew, model_info);
    else
        pred_mean = [];
        pred_mse = [];

        remain = mod(new_ns, part_num);
        if remain == 0
            n_iter = new_ns/part_num;
        else
            n_iter = (new_ns-remain)/part_num+1;
        end

        for idx = 1:n_iter
            if idx < n_iter
                [temp_pred_mean, temp_pred_mse] = calcul_mean_var(Xnew((part_num*(idx-1)+1):part_num*idx, :), model_info);
            else
                [temp_pred_mean, temp_pred_mse] = calcul_mean_var(Xnew((part_num*(idx-1)+1):end, :), model_info);
            end
            pred_mean = [pred_mean; temp_pred_mean];
            pred_mse = [pred_mse; temp_pred_mse];
        end
    end
    
    % Denormalization
    pred_mean = pred_mean*(maxY-minY) + minY;
    pred_mse = ((maxY-minY)^2).*pred_mse;
end

function [mu, mse] = calcul_mean_var(Xnew, info)
    F = info.F;
    ns = info.ns;
    normX = info.normX;
    U = info.U;
    T = info.T;
    beta_hat = info.beta_hat;
    theta = info.theta;
    var = info.var;
    func_lf = info.func_lf;

    minX = info.minX;
    maxX = info.maxX;
    minY = info.minY;
    maxY = info.maxY;

    % standardization
    normXnew = (Xnew - minX)./(maxX-minX);

    new_ns = size(normXnew, 1);
    r_U = zeros(ns, new_ns);
    for i = 1:new_ns
        r_U(:,i) = exp(-sum((theta.*((normX - repmat(normXnew(i,:), [ns, 1])).^2)),2));
    end

    R_U = zeros(new_ns, new_ns);
    for i = 1:new_ns
        R_U(:,i) = exp(-sum((theta.*((normXnew - repmat(normXnew(i,:), [new_ns, 1])).^2)),2));
    end

    if size(r_U) ~= [ns, new_ns]
        error('r_U size error')
    elseif size(R_U) ~= [new_ns, new_ns]
        error('R_U size error')
    end

    F_U = (func_lf(Xnew) - minY)./(maxY-minY);
    
    R1rU = U\(U'\r_U); % R\r_U
    FR1F = F'*(U\(U'\F)); % F'*R\F
    u_U = F_U'-F'*R1rU;
    if size(F_U, 1) ~= new_ns
        error('F_U size error')
    end

    mu = beta_hat*F_U + r_U'*(U\(U'\T)); % T = normY-G*beta_hat in krig_fit func.
    covar = var*(R_U+u_U'*(FR1F\u_U)-r_U'*(U\(U'\r_U)));
    mse = diag(covar);
end