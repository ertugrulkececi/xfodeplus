function learnable_parameters = initialize_trimf_IT2(input_data, output_data, number_of_rule)

    number_inputs = size(input_data,2);
    number_outputs = size(output_data,2);

    left = ones(1, number_inputs);
    left = dlarray(left*min(input_data));

    interval = (max(input_data) - left)/(number_of_rule - 1);
    lambdas = dlarray(repmat(interval, [number_of_rule + 1, number_inputs]));
    lambdas = log(exp(lambdas)-1);
    h = dlarray(rand(number_of_rule, number_inputs));

    learnable_parameters.leftmost_center = left;
    learnable_parameters.lambdas = lambdas;
    learnable_parameters.h = h;

    a = rand(number_of_rule*number_outputs,number_inputs)*0.01;
    learnable_parameters.linear.a = dlarray(a);

    b = rand(number_of_rule*number_outputs,1)*0.01;
    learnable_parameters.linear.b = dlarray(b);

end
