function learnable_parameters = initialize_Glorot_IT2(input_data, output_data, number_of_rule)

output_membership_type = "linear";
CSCM = "KM";

%% centers with Kmean
number_inputs = size(input_data,2);
number_outputs = size(output_data,2);

data = input_data;
data = extractdata(permute(data,[3 2 1]));

for i=1:number_inputs %applying Kmeans clustring for each input
    [~,centers(:,i)] = kmeans(data(:,i),number_of_rule);
end

learnable_parameters.input_centers = dlarray(centers);

%% sigmas

s = std(data);
s(s == 0) = 1; %to eleminate 0 initialization
s = repmat(s,number_of_rule,1);

learnable_parameters.input_sigmas = dlarray(s);

a = rand(number_of_rule*number_outputs,number_inputs)*0.01;
learnable_parameters.linear.a = dlarray(a);

b = rand(number_of_rule*number_outputs,1)*0.01;
learnable_parameters.linear.b = dlarray(b);

h = rand(number_of_rule,number_inputs);
learnable_parameters.input_h = dlarray(h);




%%
if output_membership_type == "linear"

    a = rand(number_of_rule*number_outputs,number_inputs)*0.01;
    learnable_parameters.linear.a = dlarray(a);

    b = rand(number_of_rule*number_outputs,1)*0.01;
    learnable_parameters.linear.b = dlarray(b);

    if CSCM == "BMM" || CSCM == "WNT" || CSCM == "WKM"
        alpha = (rand(1,number_outputs)*2)-1;
        learnable_parameters.linear.alpha = dlarray(alpha);
    end
end

end
