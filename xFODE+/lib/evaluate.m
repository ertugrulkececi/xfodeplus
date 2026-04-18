function [yPreds_mean, yPreds_lower, yPreds_upper] = evaluate(xTrain0, t, ux, Learnable_parameters, number_inputs, mbs, number_of_rules, number_outputs, input_membership_type, u)

x0 = xTrain0;
yPred = x0';
ahead = length(t)-1;
Ux = permute(ux, [1, 3, 2]);
X_mean = dlarray(zeros(size(x0,2) ,ahead, size(x0,1)));


for ct=1:ahead

    subnet_lowers = [];
    subnet_uppers = [];
    subnet_means = [];

    u0 = Ux(:, :, ct);
    x_ = permute(x0, [2 3 1]);
    x_u0 = permute([x_;u0], [3 1 2]);

    z0 = x_u0;

    for i = 1: number_inputs

        subnet = Learnable_parameters.("subnet" + i);
        [subnet_lower, subnet_upper, subnet_mean] = IT2_fismodel(z0(:, i, :), number_of_rules, 1, number_outputs, mbs, subnet, input_membership_type, u);
        subnet_lower = permute(subnet_lower, [1 3 2]);
        subnet_upper = permute(subnet_upper, [1 3 2]);
        subnet_mean = permute(subnet_mean, [1 3 2]);

        subnet_lowers = [subnet_lowers; subnet_lower]; 
        subnet_uppers = [subnet_uppers; subnet_upper]; 
        subnet_means = [subnet_means; subnet_mean];

    end

    dx = aggregration_output(subnet_means); 

    dx_lower = aggregration_output(subnet_lowers); 
    dx_upper = aggregration_output(subnet_uppers); 

    % State update
    x_new = x0 + dx;
    x_new = permute(x_new, [2, 1, 3]);
    X_mean(:, ct, :) = x_new;

    % PI update
    x_new_lower = x0 + dx_lower;
    x_new_lower = permute(x_new_lower, [2, 1, 3]);
    X_lower(:,ct,:) = x_new_lower;

    x_new_upper = x0 + dx_upper;
    x_new_upper = permute(x_new_upper, [2, 1, 3]);
    X_upper(:,ct,:) = x_new_upper;

    x0 = permute(x_new, [2, 1, 3]);

end

yPreds_mean = [yPred X_mean];

yPreds_lower = [yPred X_lower];
yPreds_upper = [yPred X_upper];

end

