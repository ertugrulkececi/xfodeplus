function loss = l2_frobenius_loss(yPred, targets, learnable_parameters, lambda, type)

    % Types: "l2" or "l2f"
    if type == "l2" || type == "l2UQ"
        l2_loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");
        loss = l2_loss;
    elseif type == "l2R_seperate"
        l2_loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");
        l1_term_agg1 = mean(abs(learnable_parameters.NN_agg1.Weights));
        l1_term_agg2 = mean(abs(learnable_parameters.NN_agg2.Weights));
        loss = l2_loss + 0.01*(l1_term_agg1 + l1_term_agg2);
    elseif type == "l2R_combined"
        l2_loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");
        l1_term_agg3 = mean(abs(learnable_parameters.NN_agg3.Weights(:)));
        loss = l2_loss + 0.01*l1_term_agg3;
    elseif type == "l2R" || type == "l2RUQ"
        l2_loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");
        l1_term = 0;
        if isfield(learnable_parameters, "output_layer")
            l1_term = l1_term + mean(abs(learnable_parameters.output_layer.Weights(:)));
        end
        loss = l2_loss + 0.01*l1_term;
    elseif type == "l2R_sigma"
        l2_loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");
        l1_output = mean(abs(learnable_parameters.output_layer.Weights(:)));

        % Regularization terms
        epsilon = 1e-6;
        % Initialize regularization terms
        sigma_penalty = 0;

        % Get the list of all subnets in learnable_parameters
        subnet_names = fieldnames(learnable_parameters);

        % Loop through all subnets
        for i = 1:length(subnet_names)
            subnet = learnable_parameters.(subnet_names{i});  % Access the current subnet

            % Regularize input_sigmas
            if isfield(subnet, 'input_sigmas')
                sigmas = subnet.input_sigmas;
                sigma_penalty = sigma_penalty - mean(log(sigmas + epsilon));
            end
        end

        loss = l2_loss + 0.01*l1_output + 0.001*abs(sigma_penalty);
    end

end
