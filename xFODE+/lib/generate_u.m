function M = generate_u(n)
    M = [];

    % Single 1 at the beginning
    v = zeros(1, n);
    v(1) = 1;
    M = [M; v];

    % Single 1 at the end
    v = zeros(1, n);
    v(end) = 1;
    M = [M; v];

    % Consecutive pairs of 1s
    for i = 1:(n-1)
        v = zeros(1, n);
        v(i) = 1;
        v(i+1) = 1;
        M = [M; v];
    end

    M = M';
end