function [output, right_points, left_points] = calculate_trimf_all(learnable_parameters, number_of_rules)


    l(1, :) = learnable_parameters.left(1, :);
    % l(1, :) = min(min(x) - 0.1, l(1, :));
    l(1, :) = min(-0.2, l(1, :));
    c(1, :) = l(1, :) + abs(learnable_parameters.lambdas(1, :));
    r(1, :) = c(1, :) + abs(learnable_parameters.lambdas(2, :));


    for i = 2:number_of_rules
        l(i, :) = c(i-1, :);
        c(i, :) = r(i-1, :);
        r(i, :) = c(i, :) + abs(learnable_parameters.lambdas(i+1, :));
    end

    r(end, :) = max(r(end, :), 1.2);
    right_points = r;
    left_points = l;
    % r(end, :) = max_x + eps;

    output = c;

    end