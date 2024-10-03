% Prediction of the 1st-order HDMR
function [total_mu, total_var, var_array] = HDMR1_pred(x, y0, model_cell)
    [ns, dim] = size(x);
    if nargout == 1
        total_mu = (1-dim)*y0*ones(ns,1);
        for idx = 1:dim
            pred_mu = predictor(x(:,idx), model_cell{idx});
            total_mu = total_mu + pred_mu;
        end
    elseif nargout == 2
        total_var = 0;
        total_mu = (1-dim)*y0*ones(ns,1);
        for idx = 1:dim
            [pred_mu, pred_var] = predictor(x(:,idx), model_cell{idx});
            total_mu = total_mu + pred_mu;
            total_var = total_var + pred_var;
        end
    else
        total_mu = (1-dim)*y0*ones(ns,1);
        var_array = zeros(ns, dim);
        for idx = 1:dim
            [pred_mu, pred_var] = predictor(x(:,idx), model_cell{idx});
            total_mu = total_mu + pred_mu;
            var_array(:, idx) = pred_var;
        end
        total_var = sum(var_array,2);
    end
end