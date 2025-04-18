
% Algorithm names
methods = {'Binary GA', 'Optimal Algorithm', 'Approximation Algorithm'};

% Corresponding objective values
values = [1231.35, 1451, 813];

% Numerical x positions for plotting
x = 1:length(methods);

% Create the plot
figure('Color', 'w'); % White background
plot(x, values, '-o', ...
    'LineWidth', 2.5, ...
    'MarkerSize', 10, ...
    'Color', [0.2 0.5 0.8]); % Line color

% Annotate each point with value
for i = 1:length(values)
    text(x(i), values(i) + 30, sprintf('%.2f', values(i)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 11, ...
        'FontWeight', 'bold');
end

% Axis settings
xticks(x);
xticklabels(methods);
ylabel('Objective Value');
title('GAP12 (Instance 1) - Algorithm Performance Comparison', 'FontSize', 14);

% Styling
set(gca, 'FontSize', 12);
grid on;
ylim([min(values) - 100, max(values) + 100]);
xlim([0.5, length(x) + 0.5]);
box off;


