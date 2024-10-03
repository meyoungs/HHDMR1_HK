function model_info = HK_fit(X_hf, Y_hf, func_lf, theta_lb, theta_ub)
    % INPUT
    % X_hf: high fidelity input data
    % Y_hf: high fidelity output data corresponding to X_hf
    % func_lf: low fidelity function
    % theta_lb, theta_ub: lower/upper bound of the correlation function parameters, respectively
    
    % OUTPUT
    % model_info: information for the HK prediction
    
    % Settings
    [ns, dim] = size(X_hf);
    
    theta0 = ones(1, dim);
    if nargin > 3
        lb = theta_lb;
        ub = theta_ub;
    else
        lb = 0.00001*ones(1, dim);
        ub = 100*ones(1, dim);
    end

    % Normalization
    minX = min(X_hf);
    maxX = max(X_hf);
    minY = min(Y_hf);
    maxY = max(Y_hf);
    normX = (X_hf - minX)./(maxX-minX);
    normY = (Y_hf - minY)./(maxY-minY);
    
    % Regression matrix F;
    F = (func_lf(X_hf) - minY)./(maxY-minY);

    % Write model information
    model_info.F = F;
    model_info.ns = ns;
    model_info.normX = normX;
    model_info.normY = normY;

    % MLE optimization
    options = optimoptions('patternsearch','TolFun',1e-6,'TolX',1e-6,'TolMesh',1e-6,'MaxIterations',3e5);
    [theta,~,~] = patternsearch(@(theta) likelihood(model_info, theta),theta0,[],[],[],[],lb,ub, options);
    
    % Correlation matrix / squared exponential kernel
    R = zeros(ns, ns);
    for i = 1:ns
        R(:,i) = exp(-sum((theta.*((normX - repmat(normX(i,:), [ns, 1])).^2)),2));
    end
    if size(R) ~= [ns, ns]
        error('R size error')
    end
    
    % Cholesky factorization
    R = R+(10+ns)*eps*eye(ns);
    U = chol(R);
    
    beta_hat = (F'*(U\(U'\F))) \ (F'*(U\(U'\normY)));
    T = normY-beta_hat*F;
    var = (T'*(U\(U'\T)))/ns;
    
    % Write model information
    model_info.theta = theta;
    model_info.X_hf = X_hf;
    model_info.Y_hf = Y_hf;
    model_info.R = R;
    model_info.U = U;
    model_info.T = T;
    model_info.beta_hat = beta_hat;
    model_info.theta = theta;
    model_info.var = var;
    model_info.func_lf = func_lf;
    model_info.minX = minX;
    model_info.maxX = maxX;
    model_info.minY = minY;
    model_info.maxY = maxY;
end