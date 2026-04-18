function learnable_parameters = initialize_gauss2mf_IT2(input_data, output_data, number_of_rule)

    %% centers
    number_inputs = size(input_data,2);
    number_outputs = size(output_data,2);

    data = input_data;
    data = extractdata(permute(data,[3 2 1]));

    initial_centers = min(data,[], 1);

    %% sigmas

    delta_dist = max(data,[], 1) - initial_centers;
    delta_gauss = delta_dist/(number_of_rule-1);
    sigma_gauss = delta_gauss/4;
    sigma_gauss = log(exp(sigma_gauss)-1);
    s = sigma_gauss;
    s(s == 0) = 1; %to eleminate 0 initialization
    learnable_sigmas = repmat(s,number_of_rule+1,1);

    learnable_parameters.input_sigmas = dlarray(learnable_sigmas);
    learnable_parameters.leftmost_centers = dlarray(initial_centers);

    h = rand(number_of_rule,number_inputs);
    learnable_parameters.input_h = dlarray(h);

    %% output (linear)

    a = rand(number_of_rule*number_outputs,number_inputs)*0.01;
    learnable_parameters.linear.a = dlarray(a);

    b = rand(number_of_rule*number_outputs,1)*0.01;
    learnable_parameters.linear.b = dlarray(b);

end
