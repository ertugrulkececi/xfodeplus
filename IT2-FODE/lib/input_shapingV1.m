function [X] = input_shapingV1(x, order, method,nx)

y  = x(:, 1:nx);
u  = x(:, nx+1:end);
N  = size(x,1);
ny = size(y,2);             % == nx
nu = size(u,2);             % number of input channels

% idx = (order+1):N;

% Ycols = zeros(N-order, (order+1)*nx);

idx   = (order+1:N).';      % rows for which all lags/diffs exist
nrows = numel(idx);

if method == "phase"
    Ycols = zeros(nrows, (order+1)*ny);
    Ucols = zeros(nrows, (0+1)*nu);

    Ucols(:, 1:nu) = u(idx, :);

    for j = 0:order
        Ycols(:, j*ny + (1:ny)) = y(idx - j, :);   % y(k-j)
        % Ucols(:, j*nu + (1:nu)) = u(idx - j, :);   % u(k-j)
    end
else
    % Stack 0..order differences (Δ^0 is identity)
    Ycols = zeros(nrows, (order+1)*ny);
    Ucols = zeros(nrows, (0+1)*nu);

    % 0th "difference" (the signal itself) at k
    Ycols(:, 1:ny) = y(idx, :);
    Ucols(:, 1:nu) = u(idx, :);

    yj = y; uj = u;     % running differences
    for j = 1:order
        % Compute j-th difference series (shortens by 1 each time)
        yj = yj(2:end, :) - yj(1:end-1, :);
        % uj = uj(2:end, :) - uj(1:end-1, :);

        % Align rows: valid rows for Δ^j start at (j+1)
        % Our target rows are idx (>= j+1), so use idx - j
        Ycols(:, j*ny + (1:ny)) = yj(idx - j, :);
        % Ucols(:, j*nu + (1:nu)) = uj(idx - j, :);
    end
end

% Final regressor matrix: [Y-features, U-features]
X = [Ycols, Ucols];
end

