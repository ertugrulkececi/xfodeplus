function [fuzzified_upper, fuzzified_lower] = calculate_trimf(x, learnable_parameters, number_of_rules)

if isa(x,"double")
    x = dlarray(x);
end

lambdas = log(1+ exp(learnable_parameters.lambdas));

params = [learnable_parameters.leftmost_center(1, :), lambdas(2:end-1,:)'];
T = dlarray(tril(ones(length(params))));
T = T.* params;

c = T* dlarray(ones(length(params), 1));

l = [-1e6; c(1:end-1)];
r = [c(2:end);1e6];



fuzzified_upper = custom_triangular_mf(x, l, c, r);


fuzzified_upper_ = permute(fuzzified_upper, [1 3 2]);




fuzzified_upper = permute(fuzzified_upper_, [1 3 2]);
h = dlarray(0.1+0.9*sigmoid(learnable_parameters.h));
fuzzified_lower = fuzzified_upper.*h;



end
%%

% Custom Triangular function
function output = custom_triangular_mf(x, l, c, r)

output = max(min((x-l)./(c-l), (r-x)./(r-c)), 0);

end