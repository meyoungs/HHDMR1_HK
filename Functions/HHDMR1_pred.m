% Prediction of the 1st-order Hybrid HDMR
function total_mu = HHDMR1_pred(x, y0, model_cell, gam)
    HDMR_mu = HDMR1_pred(x, y0, model_cell);
    FHDMR_mu = FHDMR1_pred(x, y0, model_cell);

    total_mu = gam*HDMR_mu + (1-gam)*FHDMR_mu;
end