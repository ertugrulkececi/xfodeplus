function [output_lower, output_upper] = T2_firing_strength_calculation_layer(lower_membership_values, upper_membership_values)

output_lower = prod(lower_membership_values, 2);
output_upper = prod(upper_membership_values, 2);

end
