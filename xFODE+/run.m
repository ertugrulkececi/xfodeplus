clc;clear;
close all;

addpath(fullfile(pwd,'lib'));
parallel.gpu.enableCUDAForwardCompatibility(true)
%% Training Options 

number_of_rules = 5;
number_of_runs = 20;
SR_method = "incremental"; 
dataset_name = "MRDamper"; %SteamEngine, HairDryer, MRDamper
[number_of_epoch, learnRate, mbs, neuralOdeTimesteps, lag] = training_prep(dataset_name);

%% Data Load

[xTrain, xTrain0, yTrue, xTest, xTest0, uTest, tTest, training_num, t, std1, mu1, ny] = data_prep(dataset_name, SR_method, lag);

number_inputs = size(xTrain,2);
number_outputs = size(yTrue, 1);

%% FLS Configuration

input_membership_type = "trimf"; % xFODE+(PS1): trimf, xFODE+(PS2): gauss2mf, xFODE+(PS3): c-gauss2mf, AFODE+: gaussmf 

if input_membership_type == "gaussmf"
    u = int2bit(0:(2^number_of_rules)-1,number_of_rules);
else
    u = generate_u(number_of_rules);
end


%% Training Loop
gradDecay = 0.9;
sqGradDecay = 0.999;

seed_list = linspace(0, number_of_runs-1, number_of_runs);
all_RMSE   = zeros(number_outputs, number_of_runs);
all_PICP   = zeros(number_outputs, number_of_runs);
all_PINAW  = zeros(number_outputs, number_of_runs);


for seed = seed_list
    rng(seed)

    clear gradients

    averageGrad = [];
    averageSqGrad = [];

    [Train] = split_data(xTrain, number_inputs, number_outputs, training_num);

    [Learnable_parameters, numer_nets, timesteps] = initialize_model(t, Train.inputs, Train.outputs, number_of_rules, input_membership_type, neuralOdeTimesteps);
    prev_learnable_parameters = Learnable_parameters;

    X = Train.inputs(:, 1:number_outputs, :);
    ux = Train.inputs(:, number_outputs+1:end, :);
    ux = permute(ux, [2 3 1]);
    ux = extractdata(ux);

    rng(seed)

    number_of_iter_per_epoch = floorDiv(training_num - neuralOdeTimesteps, mbs);

    number_of_iter = number_of_epoch * number_of_iter_per_epoch;
    global_iteration = 1;

    ind = 1;
    for epoch = 1: number_of_epoch

        [batch_inputs, U, batch_targets] = create_mini_batch(X, ux, neuralOdeTimesteps, training_num-neuralOdeTimesteps);

        batch_loss = 0;

        batch_loss_uq = 0;

        batch_loss_acc = 0;

        for iter = 1:number_of_iter_per_epoch

            [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets, iter, mbs);

            %calculating loss and gradients
            [loss, loss_uq, loss_acc, gradients, ~] = dlfeval(@IT2Loss, timesteps, mini_batch_inputs ,...
                number_inputs, u_mini_batch, targets, number_outputs, number_of_rules, mbs, Learnable_parameters, input_membership_type, u);

            % updating parameters
            [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
                epoch, learnRate, gradDecay, sqGradDecay);

            batch_loss = batch_loss + loss;

            batch_loss_uq = batch_loss_uq + loss_uq;

            batch_loss_acc = batch_loss_acc + loss_acc;

        end

        batch_loss = batch_loss/number_of_iter_per_epoch;

        batch_loss_uq = batch_loss_uq/number_of_iter_per_epoch;

        batch_loss_acc = batch_loss_acc/number_of_iter_per_epoch;

        % fprintf('Seed %d | Epoch %d | Batch_loss = %.4f | Batch_loss_UQ = %.4f | Batch_loss_ACC = %.4f\n', seed, epoch, batch_loss, batch_loss_uq, batch_loss_acc);
 
    end

    [yPreds_mean, yPreds_lower, yPreds_upper] = evaluate(xTest0, tTest, uTest, Learnable_parameters, number_inputs, mbs, number_of_rules, ...
        number_outputs, input_membership_type, u);

    yPreds_mean = yPreds_mean.*std1 + mu1;
    yTestVal = xTest.*std1 + mu1;

    err = yTestVal - yPreds_mean;

    NRMSE = sqrt(sum(err.^2, 2)./sum((yTestVal-mean(yTestVal, 2)).^2, 2));
    accuracy = 100*(1-NRMSE);

    testRMSE = rmse(yTestVal, yPreds_mean, 2);

    yPreds_lower = yPreds_lower.*std1 + mu1;
    yPreds_upper = yPreds_upper.*std1 + mu1;

    testPICP = PICP(yTestVal, yPreds_lower, yPreds_upper);
    testPINAW = PINAW(yTestVal, yPreds_lower, yPreds_upper);

    run_idx = seed + 1;
    all_RMSE(:, run_idx)     = extractdata(testRMSE);
    all_PICP(:, run_idx)     = testPICP';
    all_PINAW(:, run_idx)     = testPINAW';
end
%% Summary

fprintf('\n--- Results over %d runs (%s | %s) ---\n', ...
    number_of_runs, dataset_name, input_membership_type);
for o = 1:ny
    fprintf('Output %d | RMSE:     mean = %.4f,  std = %.4f\n', o, mean(all_RMSE(o,:)),     std(all_RMSE(o,:)));
    fprintf('Output %d | PICP:     mean = %.4f,  std = %.4f\n', o, mean(all_PICP(o,:)),     std(all_PICP(o,:)));
    fprintf('Output %d | PINAW:     mean = %.4f,  std = %.4f\n', o, mean(all_PINAW(o,:)),     std(all_PINAW(o,:)));
end
