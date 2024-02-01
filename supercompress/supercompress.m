function [P,W,adjR_squared,idx_total] = supercompress(X,Y,K,type)
% calculation of the set of compressed data points P and its correpsonding
% answers W for the (robust) supercompress algorithm
% Input: - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
%        - set of N corresponding answers Y as real numbers
%          (1xN-matrix)
%        - natural number K, dermining the number of points in P
%        - type (0 = 'original supercompress', 1 = 'robust supercompress')
% Output: - compressed data set P (sxN-matrix)
%         - corresponding answers W (1xK-matrix)
%         - (1xk-vector) adjR_squared with the adjusted R-squared value 
%           after each compression iteration
%         - (1xN-vector) idx_total with index of compressed point to which  
%           each original data point is asigned to


N = size(X,2);
s = size(X,1);
lambda = 1/(1+s);

%% initialization of first 2 clsuters
[idx_total,x_help] = kmeans(X',2); % do first clustering
P(1,:) = x_help(1,:); % save cluster centers
P(2,:) = x_help(2,:); % save cluster centers
W = zeros(K,1); % vector with average Y-value per cluster
L = zeros(K,1); % vector with loss per cluster

% calculate average Y-value and loss per cluster for both cluster choices
for k = 1:2
    W(k) = sum(Y(idx_total == k))/sum(idx_total == k); % calculate average Y-value per cluster
    if type == 0  % calculate loss per cluster
        L(k) = sum((Y(idx_total == k)-W(k)).^2);
    elseif type == 1
        L(k) = sum(norm((X(:,idx_total == k)-P(k,:)'))^2) + sum((sqrt((1-lambda)/lambda)*Y(idx_total == k)-W(k)).^2);
    else
        fprintf('The input type is not valid.')
        return
    end
end
adjR_squared = 1 - (sum(L)/(N-K))/((sum((Y-mean(Y)).^2))/(N-1)); % calculate R^2

%% loop over the remaining clusters
for k = 3:K
    [~,i_star] = max(L); % choose cluster with maximal loss
    L_prime = L;
    % if the chosen cluster only has one point, choose the next possible
    % cluster with highest L-value
    while sum(idx_total == i_star) == 1
        [~,i_star] = max(L_prime(~ismember(L_prime,max(L_prime))));
        L_prime = L_prime(~ismember(L_prime,max(L_prime)));
    end
    [idx,x_help] = kmeans(X(:,idx_total == i_star)',2); % split this cluster into 2 clusters
    idx(idx == 2) = k; % translate cluster index
    idx(idx == 1) = i_star; % translate cluster index
    P(i_star,:) = x_help(1,:); % update cluster centers
    P(k,:) = x_help(2,:); % update cluster centers
    j = 1;
    for i = 1:N
        if idx_total(i) == i_star
            idx_total(i) = idx(j); % update new cluster index for the points in the splitted cluster
            j = j+1;
        end
    end
    % calculate average Y-value and loss per cluster and the R^2
    W(i_star) = sum(Y(idx_total == i_star))/sum(idx_total == i_star);
    W(k) = sum(Y(idx_total == k))/sum(idx_total == k);
    if type == 0
        L(i_star) = sum((Y(idx_total == i_star)-W(i_star)).^2);
        L(k) = sum((Y(idx_total == k)-W(k)).^2);
    else
        L(i_star) = sum(norm((X(:,idx_total == i_star)-P(i_star,:)'))^2) + sum((sqrt((1-lambda)/lambda)*Y(idx_total == i_star)-W(i_star)).^2);
        L(k) = sum(norm((X(:,idx_total == k)-P(k,:)'))^2) + sum((sqrt((1-lambda)/lambda)*Y(idx_total == k)-W(k)).^2);
    end
    adjR_squared(k-1) = 1 - (sum(L)/(N-k))/((sum((sqrt((1-lambda)/lambda)*Y-mean(Y)).^2))/(N-1)); % calculate R^2
end

P = P';
W = W';

end