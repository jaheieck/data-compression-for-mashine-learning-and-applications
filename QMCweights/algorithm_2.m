function [N_r_i] = algorithm_2(i,r)
% calculation of N_r,i with Algorithm (2) (O(r^2s))
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

% update the values componentwise
for j = 2:s
    for r_prime = 0:r
        N_r_i(2,r_prime+1) = sum(N_r_i(1,max(r_prime+1-i(j),1):r_prime+1));
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