% Load the dataset
load('mergedData.mat');

% Separate features and targets
features = merged_data(:, 1:end-1); % Features
targets = merged_data(:, end);     % Targets 

% Handle multicollinearity by removing highly correlated features
corr_matrix = corrcoef(features); % Correlation matrix
% Find indices of redundant features (absolute correlation > 0.99)
redundant_columns = [];
for i = 1:size(corr_matrix, 1)
    for j = i+1:size(corr_matrix, 2)
        if abs(corr_matrix(i, j)) > 0.99
            redundant_columns = [redundant_columns, j]; % Collect the column indices
        end
    end
end
redundant_columns = unique(redundant_columns); % Remove duplicate indices

% Remove redundant features
if ~isempty(redundant_columns)
    fprintf('Removing %d redundant features due to high correlation.\n', length(redundant_columns));
    features(:, redundant_columns) = []; % Remove redundant features
end

% Apply PCA to reduce dimensions
[coeff, pca_features, ~, ~, explained] = pca(features); % Perform PCA

% Retain enough components to explain 95% variance
explained_variance_threshold = 95;
cumulative_explained = cumsum(explained);
num_components = find(cumulative_explained >= explained_variance_threshold, 1);

% Select reduced features
optimized_features = pca_features(:, 1:num_components);

% Normalize the reduced features
optimized_features = normalize(optimized_features, 'range');

% Transpose data for neural network
inputs = optimized_features'; % Transpose features
targets = targets';           % Transpose targets

% Initialize arrays to store validation and test accuracy for multiple iterations
val_accuracies = zeros(10, 1);
test_accuracies = zeros(10, 1);
all_results = cell(10, 1); % Store results for each iteration

% Loop for training and testing 10 times
for iteration = 1:10
    fprintf('Iteration %d of 10\n', iteration);
    
    % Create and configure the neural network
    hiddenLayerSize = [20, 10]; % Hidden layer configuration
    net = feedforwardnet(hiddenLayerSize, 'trainlm'); % Create a new NN

    % Divide data into training, validation, and test sets
    net.divideParam.trainRatio = 0.7; % 70% training
    net.divideParam.valRatio = 0.15;  % 15% validation
    net.divideParam.testRatio = 0.15; % 15% testing

    % Set training parameters
    net.trainParam.epochs = 1000;  % Max epochs
    net.trainParam.goal = 1e-6;    % Performance goal
    net.trainParam.min_grad = 1e-7; % Minimum gradient

    % Train the network with PCA-reduced inputs
    [net, tr] = train(net, inputs, targets);

    % Evaluate validation accuracy
    val_inputs = inputs(:, tr.valInd); % Validation inputs
    val_targets = targets(tr.valInd);  % Validation targets

    val_outputs = net(val_inputs); % Get predictions
    val_predicted_classes = round(val_outputs); % Round to nearest integer
    val_actual_classes = val_targets;

    % Calculate validation accuracy
    val_correct_predictions = sum(val_predicted_classes == val_actual_classes);
    val_total_samples = length(val_actual_classes);
    val_accuracy = (val_correct_predictions / val_total_samples) * 100;
    val_accuracies(iteration) = val_accuracy; % Store validation accuracy

    % Evaluate test accuracy
    test_inputs = inputs(:, tr.testInd); % Test inputs
    test_targets = targets(tr.testInd); % Test targets

    test_outputs = net(test_inputs); % Get predictions
    test_predicted_classes = round(test_outputs);
    test_actual_classes = test_targets;

    % Calculate test accuracy
    test_correct_predictions = sum(test_predicted_classes == test_actual_classes);
    test_total_samples = length(test_actual_classes);
    test_accuracy = (test_correct_predictions / test_total_samples) * 100;
    test_accuracies(iteration) = test_accuracy; % Store test accuracy

    % Store the results for each iteration
    results.val_accuracy = val_accuracy;      % Validation accuracy
    results.test_accuracy = test_accuracy;    % Test accuracy
    results.num_components = num_components;  % Number of PCA components used
    results.net = net;                        % Trained NN with PCA
    results.training_record = tr;             % Training record

    all_results{iteration} = results;

    % Save the results for the current iteration
    save(sprintf('optimized_trained_results_%d.mat', iteration), 'results');
    
    fprintf('Iteration %d: Validation Accuracy = %.2f%%, Test Accuracy = %.2f%%\n', iteration, val_accuracy, test_accuracy);
end

% Save all results
save('optimized_all_results.mat', 'val_accuracies', 'test_accuracies', 'all_results');

% Display summary results
fprintf('Average Validation Accuracy: %.2f%%\n', mean(val_accuracies));
fprintf('Average Test Accuracy: %.2f%%\n', mean(test_accuracies));

% Visualize PCA Explained Variance
figure;
pareto(explained); % Plot variance explained by each principal component
title('PCA Explained Variance');
