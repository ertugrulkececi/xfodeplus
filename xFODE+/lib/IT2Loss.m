function [loss, loss_uq, loss_acc, gradients, yPred] = IT2Loss(timesteps, x, number_inputs, u_mini_batch, targets, number_outputs, number_of_rules, mbs, learnable_parameters, input_mf_type, u)
[yPred, yPred_lower, yPred_upper] = model(timesteps, x, u_mini_batch, number_of_rules, number_inputs, number_outputs, mbs, learnable_parameters, input_mf_type, u);

loss_acc = l1loss(yPred, targets, NormalizationFactor="batch-size", DataFormat="STB");

loss_uq = tilted_loss(targets, yPred_lower, yPred_upper, 0.005, 0.995, mbs);

loss =  loss_uq + loss_acc;

gradients = dlgradient(loss, learnable_parameters);

end
