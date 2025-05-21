% Load the merged dataset
load('mergedData.mat');
% The merged data should be in the variable `merged_data`.

% Separate features and targets(labels)
% Assuming the last column is the user identifier (target/labels) 
% And other all columns correspond to the Features
inputs = merged_data(:, 1:end-1)'; % Features (transpose to match NN input format)
targets = merged_data(:, end)';   % User identifiers(targets/labels) (transpose for NN format)

% Normalize the input data (optional, improves NN performance)
inputs = normalize(inputs, 'range'); % Define Normalize to [0, 1] range

% Create and configure the neural network
hiddenLayerSize = [20,10]; % Define the size of the hidden layer
net = feedforwardnet(hiddenLayerSize, 'trainlm'); % Feedforward NN with Levenberg-Marquardt as the training algorithm

% Divide data into training, validation, and test sets
net.divideParam.trainRatio = 0.7; % 70% training
net.divideParam.valRatio = 0.15; % 15% validation
net.divideParam.testRatio = 0.15; % 15% testing

% Set training parameters (optional, can tweak to improve performance)
net.trainParam.epochs = 1000;  % Maximum number of training iterations
net.trainParam.goal = 1e-6;    % Performance goal (MSE)
net.trainParam.min_grad = 1e-7; % Minimum gradient

% Train the neural network
[net, tr] = train(net, inputs, targets);

% Evaluate intra-variance and inter-variance for the training set
% Extract training set inputs and targets
train_inputs = inputs(:, tr.trainInd);  % Features of training set
train_targets = targets(tr.trainInd);  % Targets of training set

% Get unique classes in the training set
unique_classes = unique(train_targets);

% Initialize intra-variance and inter-variance
intra_variance = 0;
inter_variance = 0;

% Calculate overall mean of the training features
overall_mean = mean(train_inputs, 2); % Overall mean across all features (column-wise)

% Total number of samples in the training set
total_samples = size(train_inputs, 2);

% use loop calculate variance metrics for each class
for i = 1:length(unique_classes)
    % Find the indices of samples belonging to the current class
    class_idx = find(train_targets == unique_classes(i));
    
    % Extract features of the current class
    class_samples = train_inputs(:, class_idx);
    
    % Number of samples in this class
    num_samples_in_class = size(class_samples, 2);
    
    % Calculate the mean of the current class
    class_mean = mean(class_samples, 2); % Mean across features for this class
    
    % Compute intra-variance (variance within the class)
    intra_variance = intra_variance + ...
        sum(sum((class_samples - class_mean).^2)) / total_samples;
    
    % Compute inter-variance (variance of class means relative to the overall mean)
    inter_variance = inter_variance + ...
        num_samples_in_class * sum((class_mean - overall_mean).^2) / total_samples;
end

% Display inter and intra variance results
fprintf('Intra-variance (Training Set): %.4f\n', intra_variance);
fprintf('Inter-variance (Training Set): %.4f\n', inter_variance);

% Evaluate training accuracy for the training set 
train_outputs = net(inputs(:, tr.trainInd)); % Network predictions for training set
train_predicted_classes = round(train_outputs); % Round predictions to nearest integer
train_actual_classes = targets(tr.trainInd);    % Actual training targets

% Calculate training accuracy
train_correct_predictions = sum(train_predicted_classes == train_actual_classes);
train_total_samples = length(train_actual_classes);
train_accuracy = (train_correct_predictions / train_total_samples) * 100;
fprintf('Training Accuracy: %.2f%%\n', train_accuracy);

% Evaluate test accuracy
test_inputs = inputs(:, tr.testInd); % Inputs for the test set
test_targets = targets(tr.testInd); % Targets for the test set

% Generate predictions for the test set
test_outputs = net(test_inputs);
test_predicted_classes = round(test_outputs);
test_actual_classes = test_targets;

% Ensure classes are positive integers starting from 1
% Shift classes to be 1-based if they start at 0 or contain negative values
min_class = min([test_actual_classes, test_predicted_classes]);
if min_class <= 0
    test_actual_classes = test_actual_classes - min_class + 1;
    test_predicted_classes = test_predicted_classes - min_class + 1;
end

% Calculate test accuracy
test_correct_predictions = sum(test_predicted_classes == test_actual_classes);
test_total_samples = length(test_actual_classes);
test_accuracy = (test_correct_predictions / test_total_samples) * 100;
fprintf('Test Set Accuracy: %.2f%%\n', test_accuracy);

% Plot performance and confusion matrix
figure;
plotperform(tr); % Performance plot (training, validation, test errors)

% One-hot encode the test targets and predicted classes for confusion matrix
num_classes = max([test_actual_classes, test_predicted_classes]); % Determine the number of unique classes
test_actual_classes_onehot = ind2vec(test_actual_classes, num_classes); % One-hot encode actual classes
test_predicted_classes_onehot = ind2vec(test_predicted_classes, num_classes); % One-hot encode predicted classes

% Plot confusion matrix using one-hot encoded labels
figure;
plotconfusion(test_actual_classes_onehot, test_predicted_classes_onehot); % Confusion matrix

% Save the trained results and model
results.train_accuracy = train_accuracy;   % Training accuracy
results.test_accuracy = test_accuracy;     % Test accuracy
results.intra_variance = intra_variance;   % Intra-variance
results.inter_variance = inter_variance;   % Inter-variance
results.net = net;                         % Trained neural network
results.training_record = tr;              % Training record (training, validation, test performance)

save('trained_results.mat', 'results');    % Save results 
fprintf('Training results saved to "trained_results.mat".\n');
