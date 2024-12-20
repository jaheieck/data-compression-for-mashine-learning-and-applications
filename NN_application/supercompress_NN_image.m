function [x,y,adjR_squared,idx_total] = supercompress_NN_image(X,Y,K,type)
% calculation of the set of compressed data points P and its correpsonding
% answers W for the (robust) supercompress algorithm for the image example
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
x(1,:) = x_help(1,:)'; % save cluster centers
x(2,:) = x_help(2,:)'; % save cluster centers
y = zeros(K,1); % vector with average Y-value per cluster
L = zeros(K,1); % vector with loss per cluster

% calculate average Y-value and loss per cluster for both cluster choices
for k = 1:2
    y(k) = sum(double(Y(idx_total == k))-1)/sum(idx_total == k); % calculate average Y-value per cluster
    if type == 0 % calculate loss per cluster
        L(k) = sum((double(Y(idx_total == k))-1-y(k)).^2); 
    elseif type == 1
        L(k) = sum(norm((X(:,idx_total == k)-x(k,:)'))^2) + sum((sqrt((1-lambda)/lambda)*(double(Y(idx_total == k))-1)-y(k)).^2); % calculate loss per cluster
    else
        fprintf('The input type is not valid.')
        return
    end
end
adjR_squared = 1 - (sum(L)/(N-2))/((sum((double(Y)-1-mean(double(Y))).^2))/(N-1)); % calculate R^2

%% loop over the remaining clusters
for k = 3:K
    [~,i_star] = max(L); % choose cluster with maximal loss
    L_prime = L;
    % if the chosen cluster only has one point, choose the next possible
    % cluster with highest L-value
    while sum(idx_total == i_star) == 1
        L_prime(i_star,1) = -1;
        [~,i_star] = max(L_prime);
    end
    [idx,x_help] = kmeans(X(:,idx_total == i_star)',2); % split this cluster into 2 clusters
    idx(idx == 2) = k; % translate cluster index
    idx(idx == 1) = i_star; % translate cluster index
    x(i_star,:) = x_help(1,:)'; % update cluster centers
    x(k,:) = x_help(2,:)'; % update cluster centers
    j = 1;
    for i = 1:N
        if idx_total(i) == i_star
            idx_total(i) = idx(j); % update new cluster index for the points in the splitted cluster
            j = j+1;
        end
    end
    % calculate average Y-value and loss per cluster and the R^2
    y(i_star) = sum(double(Y(idx_total == i_star))-1)/sum(idx_total == i_star);
    y(k) = sum(double(Y(idx_total == k))-1)/sum(idx_total == k);
    if type == 0
        L(i_star) = sum((double(Y(idx_total == i_star))-1-y(i_star)).^2);
        L(k) = sum((double(Y(idx_total == k))-1-y(k)).^2);
    else
        L(i_star) = sum(norm((X(:,idx_total == i_star)-1-x(i_star,:)'))^2) + sum((sqrt((1-lambda)/lambda)*(double(Y(idx_total == i_star))-1)-y(i_star)).^2);
        L(k) = sum(norm((X(:,idx_total == k)-x(k,:)'))^2) + sum((sqrt((1-lambda)/lambda)*(double(Y(idx_total == k))-1)-y(k)).^2);
    end
    adjR_squared(k-1) = 1 - (sum(L)/(N-k))/((sum((double(Y)-1-mean(double(Y))).^2))/(N-1));
end

x = x';
y = y';

end