% Prediction of the 1st-order FHDMR
function total_mu = FHDMR1_pred(x, y0, model_cell)
    [ns, dim] = size(x);

    total_mu = y0*ones(ns,1);
    for idx = 1:dim
        pred_mu = predictor(x(:,idx), model_cell{idx})-y0;
        ridx = pred_mu/y0;
        total_mu = total_mu.*(1+ridx);
    end
end