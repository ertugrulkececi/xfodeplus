function [output_lower, output_upper, output_mean] = IT2_fismodel(mini_batch_inputs, number_of_rule, number_inputs, number_outputs, mbs, learnable_parameters, input_membership_type, u)

    %fuzzification
    [fuzzifed_lower, fuzzifed_upper] = T2_matrix_fuzzification_layer(mini_batch_inputs, input_membership_type, learnable_parameters, number_of_rule, number_inputs, mbs);

    [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper);

    %type reduction and defuzzification
    [output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(mini_batch_inputs, firestrength_lower, firestrength_upper, learnable_parameters, number_outputs, mbs, number_of_rule, u);

end
