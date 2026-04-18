function [centers, left_sigmas, right_sigmas] = calculate_centers(leftmost_center, sigmas)

    % Number of centers (m) is one less than the length of sigmas
    m = size(sigmas, 1) - 1;
    n = size(sigmas, 2); % Number of columns for different batches/experiments
    sigmas = log(1 + exp(sigmas));

    % Calculate the left and right sigmas for each Gaussian
    left_sigmas = sigmas(1:end-1, :);
    right_sigmas = sigmas(2:end, :);


    %% Alternative center calculation

    increments = 4 * sigmas(2:end-1, :);
    params = [leftmost_center, increments'];
    T = dlarray(tril(ones(length(params))));
    T = T.* params;
    centers = T* dlarray(ones(length(params), 1));

    left_sigmas(1, :, :) = left_sigmas(1, :, :) * 1e6;    % Leftmost left sigma
    right_sigmas(end, :, :) = right_sigmas(end, :, :) * 1e6;  % Rightmost right sigma

end