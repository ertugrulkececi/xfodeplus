function [meanPINAW, stdPINAW, arrayPINAW] = resultsPINAW(number_of_seed, number_of_rule, input_membership_type)

arrayPINAW = zeros(3,number_of_seed);

% fig = figure;

% hold on

for k = 1:number_of_seed

    filename = sprintf('r%d_%s_H_KM_results_seed%d',number_of_rule, input_membership_type, k-1);

    load(filename);

    arrayPINAW(:,k) = PINAW_testFinal;

    % plot(batch_loss_log)

    % figure

    % plot_gauss2mf_omer(Learnable_parameters, 0, 0, 1, Train, nF, prev_learnable_parameters, "none", "none",dr_method)
end

meanPINAW = mean(arrayPINAW,2);
stdPINAW = std(arrayPINAW,0,2);

arrayPINAW = arrayPINAW';

grid on
ylabel("Loss")
xlabel("Epoch")

% --- Automatically save at high quality ---
% 
% match paper size to figure size
% set(fig, 'Units','Inches');
% pos = get(fig, 'Position');
% set(fig, 'PaperUnits','Inches', ...
%          'PaperPosition',[0 0 pos(3) pos(4)], ...
%          'PaperSize',[pos(3) pos(4)]);
% 
% use painters renderer for crisp vector output
% set(fig, 'Renderer','painters');
% 
% build filename
% basefn = sprintf('r%d_nF%d_trimf_softplus', number_of_rule, nF);
% 
% save as high-res PNG (300 dpi)
% print(fig, [basefn '.png'], '-dpng', '-r300');
