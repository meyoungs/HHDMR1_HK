function L = likelihood(info, theta)
    F = info.F;
    ns = info.ns;
    X = info.normX;
    Y = info.normY;

    % Correlation Matrix / squared exponential kernal
    R = zeros(ns, ns);
    for i = 1:ns
        R(:,i) = exp(-sum((theta.*((X - repmat(X(i,:), [ns, 1])).^2)),2));
    end

    % Cholesky factorization & nugget effect for numerical stability
    R = R+(10+ns)*eps*eye(ns);
    U = chol(R); % R = U'*U;

    beta_hat = (F'*(U\(U'\F))) \ (F'*(U\(U'\Y)));
    T = Y-F*beta_hat;
    var = (T'*(U\(U'\T)))/ns;
    
    L = (prod(diag(U),1).^(2/ns))*var;
end