function [X, X_lower, X_upper] = model(t, mini_batch_inputs, u_mini_batch, number_of_rule, number_inputs, number_outputs, mbs, learnable_parameters, input_membership_type, u)

x0 = mini_batch_inputs;
x = x0;

ahead = length(t)-1;
Ux = permute(u_mini_batch, [1, 3, 2]);
X = dlarray(zeros(size(x,2) ,ahead, size(x,3)));
X_lower = dlarray(zeros(size(x,2) ,ahead, size(x,3)));
X_upper = dlarray(zeros(size(x,2) ,ahead, size(x,3)));

for ct = 1:ahead
 subnet_lowers = [];
 subnet_uppers = [];
 subnet_means = [];

 u0 = Ux(:, :, ct);
 x_ = permute(x, [2 3 1]);
 x_u0 = permute([x_;u0], [3 1 2]);

 latent_space = number_inputs;
 z0 = x_u0;

 for i = 1: latent_space

     subnet = learnable_parameters.("subnet" + i);
     [subnet_lower, subnet_upper, subnet_mean] = IT2_fismodel(z0(:, i, :), number_of_rule, 1, number_outputs, mbs, subnet, input_membership_type, u);
     subnet_lower = permute(subnet_lower, [1 3 2]);
     subnet_upper = permute(subnet_upper, [1 3 2]);
     subnet_mean = permute(subnet_mean, [1 3 2]);

     subnet_lowers = [subnet_lowers; subnet_lower]; % subnet outputs are put respectively [1 2 3 4]
     subnet_uppers = [subnet_uppers; subnet_upper]; % subnet outputs are put respectively [1 2 3 4]
     subnet_means = [subnet_means; subnet_mean];

 end

 dx = aggregration_output(subnet_means); 
 dx_lower = aggregration_output(subnet_lowers); 
 dx_upper = aggregration_output(subnet_uppers); 

 % State update
 x_new = x + dx;
 x_new = permute(x_new, [2, 1, 3]);
 X(:, ct, :) = x_new;

 % PI update
 x_new_lower = x + dx_lower;
 x_new_lower = permute(x_new_lower, [2, 1, 3]);
 X_lower(:,ct,:) = x_new_lower;

 x_new_upper = x + dx_upper;
 x_new_upper = permute(x_new_upper, [2, 1, 3]);
 X_upper(:,ct,:) = x_new_upper;

 x = permute(x_new, [2, 1, 3]);

end
end

