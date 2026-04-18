function [output_lower, output_upper] = T2_matrix_fuzzification_layer(x, membership_type, learnable_parameters, number_of_rules, number_inputs, mbs)
%

input_matrix = repmat(x, number_of_rules, 1, 1);

output_upper = zeros(number_of_rules, number_inputs, mbs, "gpuArray");
output_lower = zeros(number_of_rules, number_inputs, mbs, "gpuArray");

if(membership_type == "gaussmf")

    h = 0.1 + 0.9*sigmoid(learnable_parameters.input_h);

    output_upper = custom_gaussmf(input_matrix, abs(learnable_parameters.input_sigmas), learnable_parameters.input_centers);

    output_lower = output_upper .* h;


elseif(membership_type == "gauss2mf")

    h = 0.1 + 0.9*sigmoid(learnable_parameters.input_h);

    [centers, left_sigmas, right_sigmas] = calculate_centers(learnable_parameters.leftmost_centers,learnable_parameters.input_sigmas);

    output_upper = custom_gauss2mf(input_matrix, left_sigmas, right_sigmas, centers);
    output_lower = output_upper .* h;


elseif(membership_type == "c-gauss2mf")

    h = 0.1 + 0.9*sigmoid(learnable_parameters.input_h);

    [centers, left_sigmas, right_sigmas] = calculate_centers(learnable_parameters.leftmost_centers,learnable_parameters.input_sigmas);

    output_upper = custom_c_gauss2mf(input_matrix, left_sigmas, right_sigmas, centers);
    output_lower = output_upper .* h;

elseif(membership_type == "trimf")
    
    [output_upper, output_lower] = calculate_trimf(x, learnable_parameters, number_of_rules);

end

end
%%

% Custom Gaussian function
function output = custom_gaussmf(x, s, c)
    exponent = -0.5 * ((x - c).^2 ./ s.^2);
    output = exp(exponent);
end

function output = custom_gauss2mf(x, left_sigma, right_sigma, c)
    % Custom Gaussian Membership Function with asymmetric (left-right) sigmas
    % x           -> input values [5, 4, batch size]
    % left_sigma  -> sigma to the left of the center [5, 4]
    % right_sigma -> sigma to the right of the center [5, 4]
    % c           -> center of the Gauss MF [5, 4]

    % Expand left_sigma, right_sigma, and c to match x dimensions
    left_sigma = repmat(left_sigma, 1, 1, size(x, 3));
    right_sigma = repmat(right_sigma, 1, 1, size(x, 3));
    c = repmat(c, 1, 1, size(x, 3));

    % Initialize output array
    output = dlarray(zeros(size(x)));

    % Calculate membership values based on left and right sigmas
    left_indices = x <= c;
    right_indices = x > c;

    % Left side calculation (x <= c)
    output(left_indices) = exp(-0.5 * ((x(left_indices) - c(left_indices)).^2 ./ left_sigma(left_indices).^2));

    % Right side calculation (x > c)
    output(right_indices) = exp(-0.5 * ((x(right_indices) - c(right_indices)).^2 ./ right_sigma(right_indices).^2));
end

function output = custom_c_gauss2mf(x, left_sigma, right_sigma, c)

    m = size(x,1);
    B = size(x,3);

    maskL   = (x <= c);
    gaussL = exp(-0.5 * ((x - c).^2) ./ left_sigma.^2);
    gaussR = exp(-0.5 * ((x - c).^2) ./ right_sigma.^2);
    evenAll = gaussL .* maskL + gaussR .* ~maskL;

    evenSel = mod((1:m).',2)==0;
    evenOut = evenAll .* evenSel; 

    cPrev = [-inf; c(1:end-1)];
    cNext = [c(2:end); inf];

    inLeftInt  = (cPrev < x) & (x < c);   % (c_{i-1}, c_i)
    inRightInt = (c < x)    & (x < cNext);% (c_i, c_{i+1})

    prevEven = cat(1, zeros(1,1,B,'like',x), evenAll(1:end-1,:,:)); % i-1
    nextEven = cat(1, evenAll(2:end,:,:),     zeros(1,1,B,'like',x)); % i+1

    oddRaw = inLeftInt .* (1 - prevEven) + inRightInt .* (1 - nextEven);

    oddSel = mod((1:m).',2)==1;

    oddOut = oddRaw .* oddSel;

    output = evenOut + oddOut;
end