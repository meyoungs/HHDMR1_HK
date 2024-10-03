% code for subspace sampling
function [proj_norm_x, full_proj_new_norm_x] = subsamp(proj_new_norm_x, x_cell, dmodel_cell, dimidx, mu)
    dim_x_set = x_cell{dimidx};
    sorted = sort([dim_x_set; proj_new_norm_x]);
    dim = size(x_cell, 1);
    
    full_proj_new_norm_x = mu.*ones(1, dim);
    n = size(sorted,1);
    if sorted(1) == proj_new_norm_x || sorted(n) == proj_new_norm_x
        proj_norm_x = proj_new_norm_x;
    else
        for iii = 2:n-1
            if sorted(iii) == proj_new_norm_x
                lb = sorted(iii-1);
                ub = sorted(iii+1);
                break
            end
        end
        dmodel = dmodel_cell{dimidx};
        [opt_x] = ga(@(x) calvar(x, dmodel), 1, [], [], [], [], lb, ub);
        proj_norm_x = opt_x;
    end
    full_proj_new_norm_x(dimidx) = proj_norm_x;
end

function var_ = calvar(x, dmodel)
    if size(x, 1) == 1
        [~, ~, var_] = predictor(x, dmodel);
    else
        [~, var_] = predictor(x, dmodel);
    end
    var_ = -var_;
end