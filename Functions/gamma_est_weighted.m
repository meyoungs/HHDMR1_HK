% estimation of gamma for relibility analysis using multivariate samples
function opt_gam = gamma_est_weighted(mul_x, mul_y, par, model_cell)
    [opt_gam] = fminbnd(@(gam_) obj_func(mul_y, HHDMR1_pred(mul_x, par.y0, model_cell, gam_), par.g_th), 0, 1);
end

function weighted_error = obj_func(true_y, est_y, g_th)
    squred_error = (true_y - est_y).^2;
    weights = exp(-((true_y-g_th)/g_th).^2);
    weighted_error = mean(weights.*squred_error);
end