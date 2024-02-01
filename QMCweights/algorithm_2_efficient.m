function [N_r_i] = algorithm_2_efficient(i,r)
% efficient calculation of N_r,i with moving sums (O(rs))
% Input: - vector i consisting of s non-negative, integer components
%        - natural number r, determing the norm of d
% Output: - N_r,i, the number of s-dimensional vectors with non-negative,
%           integer entries with norm r and d <= i


% initialization of the dimension and N_r_i. N_r_i needs 2 rows. The second
% saves the new (freshly calculated) values. The first the old values, 
% which are relevant for the calculation of the new values 
s = length(i);
N_r_i = zeros(2,r+1);

% execution for j=1 for both rows at the same time 
for r_prime = 0:r
    if r_prime <= i(1)
        N_r_i(:,r_prime+1) = 1;
    end
end

% update the values componentwise with "moving sums". If i(j) >= r, only 
% the first loop is relevant. r_prime=0 can be skipped, since this value
% does not change
for j = 2:s
    % if r_prime <= i(j), only N_r_i(1,r_prime+1) has to be added to the 
    % sum of r_prime-1
    for r_prime = 1:min(r,i(j))
        N_r_i(2,r_prime+1) = N_r_i(2,r_prime) + N_r_i(1,r_prime+1);
    end
    % if r_prime > i(j), only N_r_i(1,r_prime+1) has to be added to the sum
    % of r_prime-1, while N_r_i(1,r_prime-i(j)) has to be substracted
    for r_prime = i(j)+1:r
        N_r_i(2,r_prime+1) = N_r_i(2,r_prime) + N_r_i(1,r_prime+1) - N_r_i(1,r_prime-i(j));
    end
    % if the calculation of j is finished, the first row can be updated as
    % well kann die erste Zeile. This is only necessary, if we keep 
    % evaluating afterwards
    if j < s
        N_r_i(1,:) = N_r_i(2,:);
    end
end

% we are only interested in N_s,r,i, which is in the last, bottom entry
N_r_i = N_r_i(2,r+1);
        
end