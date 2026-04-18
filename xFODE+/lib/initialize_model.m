function [subnets, numer_nets, timesteps] = initialize_model(t, input_data, output_data, number_of_rule, input_membership_type, neuralOdeTimesteps)

dt = t(2) - t(1); % logical
timesteps = (0:neuralOdeTimesteps)*dt + t(1);

number_inputs = size(input_data, 2);
number_outputs = size(output_data, 2);

subnets = struct;
numer_nets = struct;


for i = 1:number_inputs

    if input_membership_type == "gauss2mf" || input_membership_type == "c-gauss2mf"
        subnet = initialize_gauss2mf_IT2(input_data(:, i, :), output_data, number_of_rule);
    elseif input_membership_type == "gaussmf"
        subnet = initialize_Glorot_IT2(input_data(:, i, :), output_data, number_of_rule);
    elseif input_membership_type == "trimf"
        subnet = initialize_trimf_IT2(input_data(:, i, :), output_data, number_of_rule);
    end
        subnets.("subnet" + i) = subnet;

    numer_nets.("numer_net" + i).subnet_mean = 0;
    numer_nets.("numer_net" + i).subnet_norm = 0;
    numer_nets.("numer_net" + i).moving_mean = 0;
    numer_nets.("numer_net" + i).moving_norm = 0;
end

subnets.("output_layer") = struct;
subnets.("output_layer").Weights = dlarray(ones(number_inputs,number_outputs));

end
