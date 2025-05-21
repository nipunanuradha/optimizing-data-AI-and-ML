% Load the trained dataset and  per-tarined model
load('mergedData.mat'); % Load the dataset
load('trained_results_10.mat'); % Load pre-trained model

% Separate features and targets from the loaded data
features = merged_data(:, 1:end-1); % Features
targets = merged_data(:, end);     % Targets/labels (user identifiers)

% Handle multicollinearity (optional)
corr_matrix = corrcoef(features); % Correlation matrix
% Find indices of redundant features
redundant_columns = [];
for i = 1:size(corr_matrix, 1)
    for j = i+1:size(corr_matrix, 2)
        if abs(corr_matrix(i, j)) > 0.99
            redundant_columns = [redundant_columns, j]; % Collect the column indices
        end
    end
end
redundant_columns = unique(redundant_columns); % Ensure unique column indices

% Remove redundant features
if ~isempty(redundant_columns)
    fprintf('Removing %d redundant features due to high correlation.\n', length(redundant_columns));
    features(:, redundant_columns) = [];
end

% Apply PCA to the dataset
[coeff, pca_features, ~, ~, explained] = pca(features); % Perform PCA

% Retain enough components to explain 95% variance
explained_variance_threshold = 95;
cumulative_explained = cumsum(explained);
num_components = find(cumulative_explained >= explained_variance_threshold, 1);

% Select reduced features
optimized_features = pca_features(:, 1:num_components);

% Normalize the reduced features
optimized_features = normalize(optimized_features, 'range');

% Transpose data for the neural network input format
inputs = optimized_features'; % Transpose features
targets = targets';           % Transpose targets

% Retrain the Neural Network with Reduced Features
hiddenLayerSize = [65]; % Hidden layer configuration
net = feedforwardnet(hiddenLayerSize, 'trainlm'); % Create a new NN

% Divide data into training, validation, and test sets
net.divideParam.trainRatio = 0.7; % 70% training
net.divideParam.valRatio = 0.15;  % 15% validation
net.divideParam.testRatio = 0.15; % 15% testing

% Set training parameters
net.trainParam.epochs = 1000;  % Maximum training iterations
net.trainParam.goal = 1e-6;    % Performance goal (MSE)
net.trainParam.min_grad = 1e-7; % Minimum gradient

% Train the network with PCA-reduced inputs
[net, tr] = train(net, inputs, targets);

% Evaluate Validation Accuracy
val_inputs = inputs(:, tr.valInd); % Validation inputs
val_targets = targets(tr.valInd);  % Validation targets

val_outputs = net(val_inputs); % Get predictions
val_predicted_classes = round(val_outputs); % Round to nearest integer
val_actual_classes = val_targets;% Actual validation labels/targets

% Calculate validation accuracy
val_correct_predictions = sum(val_predicted_classes == val_actual_classes);
val_total_samples = length(val_actual_classes);
val_accuracy = (val_correct_predictions / val_total_samples) * 100;

% Evaluate Optimization (Test) Accuracy
test_inputs = inputs(:, tr.testInd); % Test inputs
test_targets = targets(tr.testInd); % Test targets

test_outputs = net(test_inputs); % Get predictions
test_predicted_classes = round(test_outputs);
test_actual_classes = test_targets;

% Ensure all class labels are positive integers starting from 1
min_class = min([test_actual_classes, test_predicted_classes]); % Get the minimum class value
if min_class <= 0
    % Adjust the class labels to make them positive integers starting from 1
    test_actual_classes = test_actual_classes - min_class + 1;
    test_predicted_classes = test_predicted_classes - min_class + 1;
end

% Calculate optimization accuracy
test_correct_predictions = sum(test_predicted_classes == test_actual_classes);
test_total_samples = length(test_actual_classes);
test_accuracy = (test_correct_predictions / test_total_samples) * 100;

% Save the Optimized Model and Results
results.val_accuracy = val_accuracy;      % Validation accuracy
results.test_accuracy = test_accuracy;    % Test accuracy
results.num_components = num_components;  % Number of PCA components used
results.net = net;                        % Trained NN with PCA

save('optimized_trained_results_10.mat', 'results'); % Save the results

% Print Accuracy Results
fprintf('Validation Accuracy After PCA: %.2f%%\n', val_accuracy);
fprintf('Test (Optimization) Accuracy After PCA: %.2f%%\n', test_accuracy);

% Visualize PCA Results and NN Performance
figure;
pareto(explained); % Plot variance explained by each principal component
title('PCA Explained Variance');

figure;
plotperform(tr); % NN performance plot (training, validation, test errors)

% One-hot encode targets and predictions for confusion matrix
num_classes = max([test_actual_classes, test_predicted_classes]); % Total number of unique classes
test_actual_onehot = ind2vec(test_actual_classes, num_classes);    % One-hot encode actual classes
test_predicted_onehot = ind2vec(test_predicted_classes, num_classes); % One-hot encode predicted classes

% Plot confusion matrix with one-hot encoding
figure;
plotconfusion(test_actual_onehot, test_predicted_onehot);
