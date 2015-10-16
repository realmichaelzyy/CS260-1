%%%% Import Data Files
disp('Reading Train Data file')
fileID1 = fopen('./car_train.data');
A = textscan(fileID1, '%s %s %s %s %s %s %s', 'delimiter',',');
fclose(fileID1);
disp('Closed Train Data file')

disp('Reading Test Data file')
fileID2 = fopen('./car_test.data');
B = textscan(fileID2, '%s %s %s %s %s %s %s', 'delimiter',',');
fclose(fileID2);
disp('Closed Test Data file')

disp('Reading Validation Data file')
fileID3 = fopen('./car_valid.data');
C = textscan(fileID3, '%s %s %s %s %s %s %s', 'delimiter',',');
fclose(fileID3);
disp('Closed Validation Data file')

X_train = zeros(length(A{1:1}), 21);%length(unique(A{1:1})));
Y_train = A{1,7};
Y_val = unique(Y_train);

X_test = zeros(length(B{1:1}), 21);%length(unique(A{1:1})));
Y_test = B{1,7};

X_valid = zeros(length(C{1:1}), 21);%length(unique(A{1:1})));
Y_valid = C{1,7};

col_1 = {'low' 'med' 'high' 'vhigh'};
col_2 = col_1;
col_3 = {'2' '3' '4' '5more'};
col_4 = {'2' '4' 'more'};
col_5 = {'small' 'med' 'big'};
col_6 = col_1(1:3);
cols = {col_1, col_2, col_3, col_4, col_5, col_6};

disp('Loading Train/Test/Validation data to Matlab Matrice')
%%%%


%%%% Load data files into Matrix
% Load class labels for train/test/validation set to each matrix
for i = 1:length(A{1,7})
    if strcmp(Y_train{i,1}, 'unacc')
        Y_train{i,1} = 1;
    elseif strcmp(Y_train{i,1}, 'acc')
        Y_train{i,1} = 2; 
    elseif strcmp(Y_train{i,1}, 'good')
        Y_train{i,1} = 3; 
    else 
        Y_train{i,1} = 4; 
    end
end

for i = 1:length(B{1,7})
    if strcmp(Y_test{i,1}, 'unacc')
        Y_test{i,1} = 1;
    elseif strcmp(Y_test{i,1}, 'acc') 
        Y_test{i,1} = 2; 
    elseif strcmp(Y_test{i,1}, 'good') 
        Y_test{i,1} = 3; 
    else 
        Y_test{i,1} = 4; 
    end
end

for i = 1:length(C{1,7})
    if strcmp(Y_valid{i,1}, 'unacc')
        Y_valid{i,1} = 1;
    elseif strcmp(Y_valid{i,1}, 'acc') 
        Y_valid{i,1} = 2; 
    elseif strcmp(Y_valid{i,1}, 'good') 
        Y_valid{i,1} = 3; 
    else 
        Y_valid{i,1} = 4; 
    end
end

% Load features for train set to matrix
for i = 1:length(A{1,1})
    for z = 1:3
        comp_col = cols{z};
        jump = (z-1)*4;
        for j = 1:length(unique(A{1,z}))
            if strcmp(A{1,z}{i}, comp_col{1})
                X_train(i,1+jump) = 1;
            elseif strcmp(A{1,z}{i}, comp_col{2})
                X_train(i,2+jump) = 1;
            elseif strcmp(A{1,z}{i}, comp_col{3})
                X_train(i,3+jump) = 1;
            else
                X_train(i,4+jump) = 1;
            end
        end
    end
    for z = 4:6
        comp_col = cols{z};
        offset = 12;
        jump = (z-4)*3;
        for j = 1:length(unique(A{1,z}))
            if strcmp(A{1,z}{i}, comp_col{1})
                X_train(i,1+offset+jump) = 1;
            elseif strcmp(A{1,z}{i}, comp_col{2})
                X_train(i,2+offset+jump) = 1;
            else
                X_train(i,3+offset+jump) = 1;
            end
        end
    end
end

% Load features for test set to each matrix
for i = 1:length(B{1,1})
    for z = 1:3
        comp_col = cols{z};
        jump = (z-1)*4;
        for j = 1:length(unique(B{1,z}))
            if strcmp(B{1,z}{i}, comp_col{1})
                X_test(i,1+jump) = 1;
            elseif strcmp(B{1,z}{i}, comp_col{2})
                X_test(i,2+jump) = 1;
            elseif strcmp(B{1,z}{i}, comp_col{3})
                X_test(i,3+jump) = 1;
            else
                X_test(i,4+jump) = 1;
            end
        end
    end
    for z = 4:6
        comp_col = cols{z};
        offset = 12;
        jump = (z-4)*3;
        for j = 1:length(unique(B{1,z}))
            if strcmp(B{1,z}{i}, comp_col{1})
                X_test(i,1+offset+jump) = 1;
            elseif strcmp(B{1,z}{i}, comp_col{2})
                X_test(i,2+offset+jump) = 1;
            else
                X_test(i,3+offset+jump) = 1;
            end
        end
    end
end

% Load features for validation set to each matrix
for i = 1:length(C{1,1})
    for z = 1:3
        comp_col = cols{z};
        jump = (z-1)*4;
        for j = 1:length(unique(C{1,z}))
            if strcmp(C{1,z}{i}, comp_col{1})
                X_valid(i,1+jump) = 1;
            elseif strcmp(C{1,z}{i}, comp_col{2})
                X_valid(i,2+jump) = 1;
            elseif strcmp(C{1,z}{i}, comp_col{3})
                X_valid(i,3+jump) = 1;
            else
                X_valid(i,4+jump) = 1;
            end
        end
    end
    for z = 4:6
        comp_col = cols{z};
        offset = 12;
        jump = (z-4)*3;
        for j = 1:length(unique(C{1,z}))
            if strcmp(C{1,z}{i}, comp_col{1})
                X_valid(i,1+offset+jump) = 1;
            elseif strcmp(C{1,z}{i}, comp_col{2})
                X_valid(i,2+offset+jump) = 1;
            else
                X_valid(i,3+offset+jump) = 1;
            end
        end
    end
end
disp('Finished Train/Test/Validation data to Matlab Matrice')
%%%%

%%%% Question 4C
new_accu = zeros(23,1);
train_accu = zeros(23,1);
for k = [1:2:23]
    [new_accu(k), train_accu(k)] = knn_classify(X_train, Y_train, X_valid, Y_valid, k);
end

for k = [1:2:23]
    [new_accu(k), train_accu(k)] = knn_classify(X_train, Y_train, X_test, Y_test, k);
end


% Based on train_accu & new_accu, run KNN with 3 different k values (7, 9 and 11)
% [test_accu_k_7, train_accu_k_7] = knn_classify(X_train, Y_train, X_test, Y_test, 7);
% [test_accu_k_9, train_accu_k_9] = knn_classify(X_train, Y_train, X_test, Y_test, 9);
% [test_accu_k_11, train_accu_k_11] = knn_classify(X_train, Y_train, X_test, Y_test, 11);
%%%%


%%%% Question 4D
D = load('./boundary.mat');

% Generate uniforma data to use as test data as well as to draw decision
% boundaries
grid_data = [];
for i =0.01:0.01:1
    for j = 0.01:0.01:1
        grid_data = [grid_data; i,j];
    end
end
boundary_features = D.features;
boundary_label = D.labels;

% Run another KNN function in order to get predicted label
Y_predict_new_k_1 = knn_classify_boundary(boundary_features, boundary_label, grid_data, 1);
Y_predict_new_k_5 = knn_classify_boundary(boundary_features, boundary_label, grid_data, 5);
Y_predict_new_k_15 = knn_classify_boundary(boundary_features, boundary_label, grid_data, 15);
Y_predict_new_k_20 = knn_classify_boundary(boundary_features, boundary_label, grid_data, 20);

% Draw decision boundary based on predicted label
gscatter(grid_data(:,1),grid_data(:,2),Y_predict_new_k_1);
gscatter(grid_data(:,1),grid_data(:,2),Y_predict_new_k_5);
gscatter(grid_data(:,1),grid_data(:,2),Y_predict_new_k_15);
gscatter(grid_data(:,1),grid_data(:,2),Y_predict_new_k_20);