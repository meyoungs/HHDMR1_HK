% 2D nonlinear function
% reference: Bichon, B. J., Eldred, M. S., Swiler, L. P., Mahadevan, S., &
% McFarland, J. M. (2008). Efficient global reliability analysis for
% nonlinear implicit performance functions. AIAA journal, 46(10),
% 2459-2468.
function y = prob_2D(x)
    % x: model input matrix, ns*dim matrix
    % y: model output vector, ns*1 vector
    y = (x(:,1).^2+4).*(x(:,2)-1)/20 - sin(5*x(:,1)/2);
end
% g_th = 2