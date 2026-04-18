function [output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(x,lower_firing_strength,upper_firing_strength, learnable_parameters, number_outputs, mbs, number_of_rules, u)

temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
x = permute(x,[2 1 3]);
temp_input = [x; ones(1, size(x, 2), size(x, 3))];
temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

c = temp_mf*temp_input;
c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

delta_f = upper_firing_strength - lower_firing_strength;
delta_f = permute(delta_f,[3 1 2]);

payda2 = delta_f*u;

%         pay2 = pagemtimes((permute(c,[3 1 2]).*repmat(delta_f,1,1,number_outputs)),u);
pay2 = pagemtimes((permute(c,[3 1 2]).*delta_f),u);

pay2 = permute(pay2,[3,2,1]);
pay1 = permute(sum(c .* lower_firing_strength,1),[2 1 3]);

pay = pay1 + pay2;

%         clear pay1_upper pay2_upper
%         clear delta_f u

payda2 = permute(payda2,[3,2,1]);
payda1 = sum(lower_firing_strength,1);

payda = payda1 + payda2;

%         clear payda1 payda2

output = pay./payda;

%         clear pay_lower pay_upper payda

output_lower = permute(min(output,[],2),[2 1 3]);
output_upper = permute(max(output,[],2),[2 1 3]);

%         clear output_lower_temp output_upper_temp

output_mean = (output_lower + output_upper)./2;



output_lower = dlarray(output_lower);
output_upper = dlarray(output_upper);
output_mean = dlarray(output_mean);

end