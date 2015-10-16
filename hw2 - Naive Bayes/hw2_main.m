[X_train, Y_train, X_test, Y_test, X_valid, Y_valid] = load_data();

% [valid_accu, train_accu] = naive_bayes(X_train, Y_train, X_valid, Y_valid);
[valid_accu, ~] = naive_bayes(X_train, Y_train, X_valid, Y_valid);
[test_accu, train_accu] = naive_bayes(X_train, Y_train, X_test, Y_test);

% Decision Tree codes (20 cases)
% Trees from train data
[row, col] = size(X_train);
[row_new, col_new] = size(X_valid);
tree_train = {};
tree_valid = {};
tree_test = {};
Y_dc_train = {};
Y_dc_valid = {};
Y_dc_test = {};

for k=1:10
    tree_train{k,1} = fitctree(X_train, Y_train, 'MinLeafSize', k ,'SplitCriterion','gdi', 'Prune','off'); % deviance    
    Y_dc_train{k,1} = predict(tree_train{k,1}, X_train);
    Y_dc_valid{k,1} = predict(tree_train{k,1}, X_valid);
    Y_dc_test{k,1} = predict(tree_train{k,1}, X_test);
end
for k=1:10
    tree_train{k,2} = fitctree(X_train, Y_train, 'MinLeafSize', k ,'SplitCriterion','deviance', 'Prune','off');
    Y_dc_train{k,2} = predict(tree_train{k,2}, X_train);
    Y_dc_valid{k,2} = predict(tree_train{k,2}, X_valid);
    Y_dc_test{k,2} = predict(tree_train{k,2}, X_test);
end

% Compute Accuracy
dtree_train_accu = zeros(2,10);
for k = 1:10
    for j = 1:row
        if Y_train(j) == Y_dc_train{k,1}(j)%, Y_train{smallest_indice(i,1)}
            dtree_train_accu(1,k) = dtree_train_accu(1, k) + 1;
        end
        if Y_train(j) == Y_dc_train{k,2}(j)
            dtree_train_accu(2,k) = dtree_train_accu(2, k) + 1;
        end
    end
    dtree_train_accu(1, k) = dtree_train_accu(1, k) / row;
    dtree_train_accu(2, k) = dtree_train_accu(2, k) / row;
end

% Compute Valid/Test Accuracy
dtree_valid_accu = zeros(2,10);
dtree_test_accu = zeros(2,10);
for k = 1:10
    for j = 1:row_new
        if Y_valid(j) == Y_dc_valid{k,1}(j)
            dtree_valid_accu(1,k) = dtree_valid_accu(1, k) + 1;
        end
        if Y_valid(j) == Y_dc_valid{k,2}(j)
            dtree_valid_accu(2,k) = dtree_valid_accu(2, k) + 1;
        end
        
        if Y_test(j) == Y_dc_test{k,1}(j)
            dtree_test_accu(1,k) = dtree_test_accu(1, k) + 1;
        end
        if Y_test(j) == Y_dc_test{k,2}(j)
            dtree_test_accu(2,k) = dtree_test_accu(2, k) + 1;
        end
    end
    dtree_valid_accu(1, k) = dtree_valid_accu(1, k) / row_new;
    dtree_valid_accu(2, k) = dtree_valid_accu(2, k) / row_new;
    
    dtree_test_accu(1, k) = dtree_test_accu(1, k) / row_new;
    dtree_test_accu(2, k) = dtree_test_accu(2, k) / row_new;
end


% 
% Logistic Regression codes 
% [B_train, dev_train, stats_train] = mnrfit(X_train,Y_train,'Model','hierarchical');
% yhat_train = mnrval(B_train,X_train,950);
B = mnrfit(X_train,Y_train); 
pihat = mnrval(B,X_train);
pihat_valid = mnrval(B,X_valid);
pihat_test = mnrval(B,X_test);

Y_predict_from_lr = zeros(row,1);
for i =1:row
    Y_predict_from_lr(i,1) = find(pihat(i,:) == max(pihat(i,:)));
end

% Compute Train Accuracy
lr_train_accu = 0.0;
for i = 1:row
    if Y_train(i) == Y_predict_from_lr(i)%, Y_train{smallest_indice(i,1)}
        lr_train_accu = lr_train_accu + 1;
    end
end
lr_train_accu = lr_train_accu / row;

Y_predict_from_lr_valid = zeros(row_new,1);
for i =1:row_new
    Y_predict_from_lr_valid(i,1) = find(pihat_valid(i,:) == max(pihat_valid(i,:)));
end

% Compute Valid Accuracy
lr_valid_accu = 0.0;
for i = 1:row_new
    if Y_valid(i) == Y_predict_from_lr_valid(i)%, Y_train{smallest_indice(i,1)}
        lr_valid_accu = lr_valid_accu + 1;
    end
end
lr_valid_accu = lr_valid_accu / row_new;

Y_predict_from_lr_test = zeros(row_new,1);
for i =1:row_new
    Y_predict_from_lr_test(i,1) = find(pihat_test(i,:) == max(pihat_test(i,:)));
end

% Compute Test Accuracy
lr_test_accu = 0.0;
for i = 1:row_new
    if Y_test(i) == Y_predict_from_lr_test(i)%, Y_train{smallest_indice(i,1)}
        lr_test_accu = lr_test_accu + 1;
    end
end
lr_test_accu = lr_test_accu / row_new;