function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data 
%
% CS260 2015 Fall, Homework 2
class_label = unique(train_label);
[row, col] = size(train_data);

class_label_indice = cell(1,4);     % Find class label index based on train label data
prior = zeros(1,4);                 % Compute prior probability
X_train_each_class = cell(1,4);     % Get feature matrix for each class label
occurences = zeros(4,21);

% likelihood ?? ?????? - ??? / ????? ??
for k = 1:length(class_label)
    class_label_indice{1,k} = find(train_label == k);
    prior(k) = length(class_label_indice{1,k}) / length(train_label);
    X_train_each_class{1,k} = train_data(class_label_indice{1,k},:);
    for i = 1:col
        occurences(k,i) = length(find(X_train_each_class{1,k}(:,i) == 1));
        likelihood(k,i) = occurences(k,i) / length(class_label_indice{1,k}); % ???% might be wrong this part
    end
end

for k = 1:length(class_label)
    for i = 1:col
        if likelihood(k,i) == 0
            likelihood(k,i) = 0.001;
        end
    end
end

probs_by_label = ones(950,4);
for i = 1:row
    for j = 1:col
        for k = 1:length(class_label)
            
            p_ki = likelihood(k,j);
            p = p_ki.^train_data(i,j);
            
            minus_p_ki = 1 - p_ki; 
            minus_p = minus_p_ki.^(1-train_data(i,j));
            
            result = p * minus_p; % might be wrong % prior(k) * 
            probs_by_label(i,k) = probs_by_label(i,k) * result; % + ??? log???
        end
    end
end

Y_predict_from_train = zeros(950,1);
for i =1:row
    Y_predict_from_train(i,1) = find(probs_by_label(i,:) == max(probs_by_label(i,:)));
end

% Compute Training Accuracy
train_accu = 0.0;
for i = 1:row
    if train_label(i) == Y_predict_from_train(i)%, Y_train{smallest_indice(i,1)}
        train_accu = train_accu + 1;
    end
end
train_accu = train_accu / row;


% New DATA
[row_new, col_new] = size(new_data);
probs_new_by_label = ones(row_new,4);
for i = 1:row_new
    for j = 1:col_new
        for k = 1:length(class_label)
            
            p_ki_new = likelihood(k,j);
            p_new = p_ki_new.^new_data(i,j);
            
            minus_p_ki_new = 1 - p_ki_new; 
            minus_p_new = minus_p_ki_new.^(1-new_data(i,j));
            
            result_new = p_new * minus_p_new;
            probs_new_by_label(i,k) = probs_new_by_label(i,k) * result_new;
        end
    end
end

Y_predict_from_new = zeros(row_new,1);
for i =1:row_new
    Y_predict_from_new(i,1) = find(probs_new_by_label(i,:) == max(probs_new_by_label(i,:)));
end

% Compute New Accuracy
new_accu = 0.0;
for i = 1:row_new
    if new_label(i) == Y_predict_from_new(i)%, Y_train{smallest_indice(i,1)}
        new_accu = new_accu + 1;
    end
end
new_accu = new_accu / row_new;