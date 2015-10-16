function [Y_predict_new] = knn_classify_boundary(boundary_features, boundary_label, grid_data, k)
% k-nearest neighbor classifier
% Input:
%  boundary_features: N*D matrix, each row as a sample and each column as a
%  feature
%  boundary_label: N*1 vector, each row as a label
%  grid_data: M*D matrix, each row as a sample and each column as a
%  feature
%  k: number of nearest neighbors
%
% Output:
%  Y_predict_new: predicted label
%
% CS260 2015 Fall, Homework 1

dist = zeros(length(boundary_features),length(boundary_features));
for i=1:length(dist)
    for j=1:length(dist)
        dist(i,j) = pdist2(boundary_features(i,:), boundary_features(j,:));
    end
end
dist(logical(eye(size(dist)))) = Inf; 

% Find k-nearest neighbors
smallest_value = [];
smallest_indice = [];
num_class_label = [];

% Predict y label based on k-nearest neighbors
Y_predict = zeros(400,1);
for i = 1:length(dist)
    for j = 1:k
        smallest_val = min(dist(:,i));
        find_smallest_indice = find(smallest_val == dist(:,i));
        index = randperm(length(find_smallest_indice));
        smallest_idx = find_smallest_indice(index(1:1));
        smallest_value(i,j) = smallest_val;
        smallest_indice(i,j) = smallest_idx;
        num_class_label(i,j) = boundary_label(smallest_idx);
        dist(smallest_idx,i) = Inf;
    end
    
    % Predict y label based on k-nearest neighbors
    [num_occurence, vec_class] = histc(num_class_label(i,:), unique(num_class_label(:,:))); 
    class_indice = find(max(num_occurence) == num_occurence);        
    index_class = randperm(length(class_indice));
    class_label = class_indice(index_class(1:1));
    Y_predict(i) = class_label;
end

dist_new = zeros(length(boundary_features), length(grid_data)); %X_test
[row, col] = size(dist_new);
for i=1:row
    for j=1:col
        dist_new(i,j) = pdist2(boundary_features(i,:), grid_data(j,:));
    end
end

% Find k-nearest neighbors
smallest_value_new = [];
smallest_indice_new = [];
num_class_label_new = [];

% Predict y label based on k-nearest neighbors
Y_predict_new = zeros(col,1);
for i = 1:col
    for j = 1:k
        smallest_val_new = min(dist_new(:,i));
        find_smallest_indice_new = find(smallest_val_new == dist_new(:,i));
        index_new = randperm(length(find_smallest_indice_new));
        smallest_idx_new = find_smallest_indice_new(index_new(1:1));
        smallest_value_new(i,j) = smallest_val_new;
        smallest_indice_new(i,j) = smallest_idx_new;
        num_class_label_new(i,j) = boundary_label(smallest_idx_new); %important point: should pick label from train data
        dist_new(smallest_idx_new, i) = Inf;
    end
    
    % Predict y label based on k-nearest neighbors
    [num_occurence_new, vec_class_new] = histc(num_class_label_new(i,:), unique(num_class_label_new(:,:))); 
    class_indice_new = find(max(num_occurence_new) == num_occurence_new);
    index_class_new = randperm(length(class_indice_new));
    class_label_new = class_indice_new(index_class_new(1:1));
    Y_predict_new(i) = class_label_new;
end
disp('Finished running KNN Boundary function')