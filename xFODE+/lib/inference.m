function [TrainLoss, valLoss, TestLoss, train_RMSE, val_RMSE, test_RMSE, yPred_train_lower, yPred_val_lower, yPred_test_lower, yPred_train_upper, yPred_val_upper, yPred_test_upper, PI_train, PI_val, PI_test, PI_NAW_train, PI_NAW_val, PI_NAW_test] = inference(Train, Val, Test, number_inputs, number_outputs, number_of_rules, Learnable_parameters, input_membership_type, u, yTrue_train, yTrue_val, yTrue_test, loss_type, lambda, y_minimum, y_range)

    [TrainLoss, yTrain_lower, yTrain_upper] = evaluate(Train.inputs, number_inputs, Train.outputs, number_outputs, number_of_rules, length(Train.inputs), Learnable_parameters, input_membership_type, u, loss_type, lambda);
    [valLoss ,yVal_lower, yVal_upper] = evaluate(Val.inputs, number_inputs, Val.outputs, number_outputs, number_of_rules, length(Val.inputs), Learnable_parameters, input_membership_type, u, loss_type, lambda);
    [TestLoss ,yTest_lower, yTest_upper] = evaluate(Test.inputs, number_inputs, Test.outputs, number_outputs, number_of_rules, length(Test.inputs), Learnable_parameters, input_membership_type, u, loss_type, lambda);


    yPred_train_upper = reshape(yTrain_upper, [number_outputs, size(Train.inputs,3)]);
    yPred_train_lower = reshape(yTrain_lower, [number_outputs, size(Train.inputs,3)]);

    yPred_val_upper = reshape(yVal_upper, [number_outputs, size(Val.inputs,3)]);
    yPred_val_lower = reshape(yVal_lower, [number_outputs, size(Val.inputs,3)]);

    yPred_test_upper = reshape(yTest_upper, [number_outputs, size(Test.inputs,3)]);
    yPred_test_lower = reshape(yTest_lower, [number_outputs, size(Test.inputs,3)]);


    PI_train = PICP(yTrue_train, yPred_train_lower, yPred_train_upper);
    PI_val = PICP(yTrue_val, yPred_val_lower, yPred_val_upper);
    PI_test = PICP(yTrue_test, yPred_test_lower, yPred_test_upper);

    PI_NAW_train = PINAW(yTrue_train, yPred_train_lower, yPred_train_upper);
    PI_NAW_val = PINAW(yTrue_val, yPred_val_lower, yPred_val_upper);
    PI_NAW_test = PINAW(yTrue_test, yPred_test_lower, yPred_test_upper);

    yPred_train = (yPred_train_upper + yPred_train_lower)/2;
    yPred_val = (yPred_val_upper + yPred_val_lower)/2;
    yPred_test = (yPred_test_upper + yPred_test_lower)/2;

    yPred_Train_denorm = max_min_denorm(yPred_train, [y_minimum], [y_range]);
    yGT_Train_denorm = max_min_denorm(Train.outputs, [y_minimum], [y_range]);

    yPred_Test_denorm = max_min_denorm(yPred_test, [y_minimum], [y_range]);
    yGT_Test_denorm = max_min_denorm(Test.outputs, [y_minimum], [y_range]);

    yGT_Train_denorm = permute(yGT_Train_denorm, [2 3 1]);
    yGT_Test_denorm = permute(yGT_Test_denorm, [2 3 1]);

    denorm_train_RMSE = rmse(yPred_Train_denorm, yGT_Train_denorm);
    denorm_test_RMSE = rmse(yPred_Test_denorm, yGT_Test_denorm);

    train_RMSE = rmse(yPred_train, yTrue_train);
    val_RMSE = rmse(yPred_val, yTrue_val);
    test_RMSE = rmse(yPred_test, yTrue_test);

    train_RMSE = denorm_train_RMSE;
    test_RMSE = denorm_test_RMSE;

end