function [meanRMSE, stdRMSE, arrayRMSE] = results(number_of_seed, number_of_rule, input_membership_type)

arrayRMSE = zeros(3,number_of_seed);

figure

hold on

for k = 1:number_of_seed

    filename = sprintf('r%d_%s_results_seed%d', number_of_rule, input_membership_type, k-1);

    load(filename);

    arrayRMSE(:,k) = testRMSE;

    plot(batch_loss_log)
end

meanRMSE = mean(arrayRMSE,2);
stdRMSE = std(arrayRMSE,0,2);

grid on
