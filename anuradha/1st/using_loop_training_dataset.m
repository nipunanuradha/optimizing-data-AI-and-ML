% Initialize an array to store results for each iteration
train_accuracies = zeros(10, 1); % Store training accuracies
test_accuracies = zeros(10, 1);  % Store test accuracies
all_results = cell(10, 1);       % Store results for each iteration

% Run the training and evaluation 10 times
for i = 1:10
    fprintf('Iteration %d of 10\n', i);
    
    % Load the merged dataset
    load('mergedData.mat');
    
    % Separate features and targets
    inputs = merged_data(:, 1:end-1)'; % Features (transpose to match NN input format)
    targets = merged_data(:, end)';    % User identifiers/targets (transpose for NN format)
    
    % Normalize the input data
    inputs = normalize(inputs, 'range'); % Normalize to [0, 1] range
    
    % Create and configure the neural network
    hiddenLayerSize = [30, 15]; % Define hidden lauer size
    net = feedforwardnet(hiddenLayerSize, 'trainlm'); % Feedforward NN with Levenberg-Marquardt algorithm
    
    % Divide data into training, validation, and test sets
    net.divideParam.trainRatio = 0.6; % 60% training
    net.divideParam.valRatio = 0.2;  % 20% validation
    net.divideParam.testRatio = 0.2; % 20% testing
    
    % Set training parameters
    net.trainParam.epochs = 500;  % Maximum number of epochs
    net.trainParam.goal = 1e-5;    % Performance goal (MSE)
    net.trainParam.min_grad = 1e-8; % Minimum gradient
    
    % Train the neural network
    [net, tr] = train(net, inputs, targets);
    
    % Evaluate training accuracy
    train_outputs = net(inputs(:, tr.trainInd)); % Network predictions for training set
    train_predicted_classes = round(train_outputs); % Round predictions to nearest integer
    train_actual_classes = targets(tr.trainInd);    % Actual training targets
    
    % Calculate training accuracy
    train_correct_predictions = sum(train_predicted_classes == train_actual_classes);
    train_total_samples = length(train_actual_classes);
    train_accuracy = (train_correct_predictions / train_total_samples) * 100;
    train_accuracies(i) = train_accuracy; % Store training accuracy
    
    % Evaluate test accuracy
    test_inputs = inputs(:, tr.testInd); % Inputs for the test set
    test_targets = targets(tr.testInd); % Targets for the test set
    
    % Generate predictions for the test set
    test_outputs = net(test_inputs);
    test_predicted_classes = round(test_outputs);
    test_actual_classes = test_targets;
    
    % Calculate test accuracy
    test_correct_predictions = sum(test_predicted_classes == test_actual_classes);
    test_total_samples = length(test_actual_classes);
    test_accuracy = (test_correct_predictions / test_total_samples) * 100;
    test_accuracies(i) = test_accuracy; % Store test accuracy
    
    % Store the results for each iteration
    results.train_accuracy = train_accuracy;   % Training accuracy
    results.test_accuracy = test_accuracy;     % Test accuracy
    results.net = net;                         % Trained neural network
    results.training_record = tr;              % Training record (training, validation, test performance)
    
    % Save the results for each iteration
    save(sprintf('trained_results_%d.mat', i), 'results');    % Save results for each iteration
    
    fprintf('Iteration %d: Training Accuracy = %.2f%%, Test Accuracy = %.2f%%\n', i, train_accuracy, test_accuracy);
end

% Save all iteration results
save('all_results.mat', 'train_accuracies', 'test_accuracies', 'all_results');

% Print final message 
fprintf('Training and testing completed for 10 iterations. Results saved.\n');
