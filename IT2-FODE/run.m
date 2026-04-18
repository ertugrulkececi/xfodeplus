clear;clc

addpath(fullfile(pwd,'lib'));
parallel.gpu.enableCUDAForwardCompatibility(true)
%% Training Options 

number_mf = 5;
number_of_runs = 20;
SR_method = "incremental";
loss_type = "L2R";
dataset_name = "HairDryer";
[number_of_epoch, learnRate, mbs, neuralOdeTimesteps,lag] = training_prep(dataset_name);

%% Data Load

[xTrain, xTrain0, yTrue, xTest, xTest0, uTest, tTest, training_num, t, std1, mu1, ny] = data_prep(dataset_name, SR_method, lag);

number_inputs = size(xTrain,2);
number_outputs = size(yTrue, 1);

%% For reproducability

input_membership_type = "gaussmf";

input_type ="H";

output_membership_type = "linear";
type_reduction_method = "KM";

gradDecay = 0.9;
sqGradDecay = 0.999;

plotFrequency = 100;

close all

if type_reduction_method == "KM"
    u = int2bit(0:(2^number_mf)-1,number_mf);
else
    u = 0;
end

%% Training Loop
seed_list = linspace(0, number_of_runs-1, number_of_runs);
all_RMSE   = zeros(number_outputs, number_of_runs);
all_PICP   = zeros(number_outputs, number_of_runs);
all_PINAW  = zeros(number_outputs, number_of_runs);

for seed = seed_list

    clear gradients

    averageGrad = [];
    averageSqGrad = [];

    rng(seed)

    % split by number ------------------------------
    data = [xTrain];
    data_size = size(data,1);
    test_num = data_size-training_num;


    Training_temp = data((1:training_num),:);

    % ------------------------------

    %training data
    Train.inputs = reshape(Training_temp(:,1:number_inputs)', [1, number_inputs, training_num]); % traspose come from the working mechanism of the reshape, so it is a must
    Train.outputs = reshape(Training_temp(:,1:number_outputs)', [1, number_outputs, training_num]);

    Train.inputs = dlarray(Train.inputs);
    % init

    [Learnable_parameters, timesteps] = initialize_Glorot_IT2_dlode(Train.inputs, number_inputs, number_outputs, input_type,output_membership_type, number_mf, type_reduction_method, t, neuralOdeTimesteps);
    prev_learnable_parameters = Learnable_parameters;

    % split data state and input

    X = Train.inputs(:, 1:number_outputs, :);
    ux = Train.inputs(:, number_outputs+1:end, :);
    ux = permute(ux, [2 3 1]);
    ux = extractdata(ux);

    % rng reset
    rng(seed)


    number_of_iter_per_epoch = floorDiv(training_num-neuralOdeTimesteps, mbs);

    number_of_iter = number_of_epoch * number_of_iter_per_epoch;
    global_iteration = 1;

    ind = 1;
    for epoch = 1: number_of_epoch

        [batch_inputs, U, batch_targets] = create_mini_batch(X, ux, neuralOdeTimesteps, training_num-neuralOdeTimesteps);
    
        batch_loss = 0;

        batch_loss_uq = 0;

        batch_loss_acc = 0;
        %
        for iter = 1:number_of_iter_per_epoch

            [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets, iter, mbs);
            [loss, loss_acc, loss_uq, gradients, yPred_train_lower, yPred_train_upper, yPred_train] = dlfeval(@IT2_ModelLoss_dlode, timesteps, mini_batch_inputs ,...
                number_inputs, u_mini_batch, targets,number_outputs, number_mf, mbs, Learnable_parameters, output_membership_type,...
                input_membership_type,input_type,type_reduction_method,u, neuralOdeTimesteps);

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

    x = xTest0;
    yPred = x';
    ahead = length(tTest)-1;

    PP = griddedInterpolant(tTest, permute(uTest,[2, 1, 3]), "pchip");
    Ux = permute(PP(tTest(:)),[2 3 1]);
    X_lower = dlarray(zeros(size(x,2) ,ahead, size(x,1)));
    X_upper = dlarray(zeros(size(x,2) ,ahead, size(x,1)));
    X_mean = dlarray(zeros(size(x,2) ,ahead, size(x,1)));

    for ct = 1:ahead
        u_mini_batch = Ux(:, :, ct);
        [x_l, x_u, x_m] = evaluatemodel(x, u_mini_batch, Learnable_parameters,number_mf, number_inputs,number_outputs, mbs, output_membership_type, input_membership_type, input_type, type_reduction_method, u);
        X_lower(:, ct) = x_l';
        X_upper(:, ct) = x_u';
        X_mean(:, ct) = x_m';
        x = x_m;
    end

    yPreds_mean = [yPred X_mean];
    yPreds_lower = [yPred X_lower];
    yPreds_upper = [yPred X_upper];

    yPreds_lower = yPreds_lower.*std1 + mu1;
    yPreds_upper = yPreds_upper.*std1 + mu1;
    yPreds_mean = yPreds_mean.*std1 + mu1;
    yTestVal = xTest.*std1 + mu1;

    err = yTestVal - yPreds_mean;

    NRMSE = sqrt(sum(err.^2, 2)./sum((yTestVal-mean(yTestVal, 2)).^2, 2));
    accuracyFinal = 100*(1-NRMSE);


    testRMSE = rmse(yTestVal, yPreds_mean, 2);

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



%%
function [X0, U, targets]  = create_mini_batch(X, ux, ahead, numexamples)

X = permute(X, [2, 3, 1]);


shuffle_idx = randperm(size(X, 2)-ahead);

X0 = dlarray(X(:, shuffle_idx));
targets = dlarray(zeros([size(X, 1) ahead, numexamples]));
U = zeros([size(ux, 1), ahead+1, numexamples]);

for i =1:numexamples
    targets(:, :, i) = X(:, shuffle_idx(i) + 1: shuffle_idx(i) + ahead);
    U(:, :, i) = ux(:, shuffle_idx(i): shuffle_idx(i) + ahead);
end

X0 = permute(X0, [3 1 2]);

end

%%
function [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets,iter,mbs)

mini_batch_inputs = batch_inputs(:, :, ((iter-1)*mbs)+1:(iter*mbs));
targets = batch_targets(:, :, ((iter-1)*mbs)+1:(iter*mbs));
u_mini_batch = U(:, :, ((iter-1)*mbs)+1:(iter*mbs));

end