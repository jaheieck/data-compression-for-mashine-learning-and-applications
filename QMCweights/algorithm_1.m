function [ST_r] = algorithm_1(z,X,Y,b,r)
% calculation of S_r and T_r based on Algorithm (1) (O(rsN))
% Input: - column vector z in [0,1)^s
%        - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
%        - set of N corresponding answers Y as real numbers
%          (1xN-matrix), where Y = ones(1,N) for the calculation of S_r
%        - prime number b for the b-adic expansion
%        - natural number r, determing the norm of d
% Output: - S_r or T_r, the relevant sum for the calculation of the weights


% define and initialize the variables
N = length(Y);
s = length(z);
ST_r = 0;

% As in Algorithm (1) suggested, we visit every point in X componentwise
for n = 1:N
    i = zeros(1,s);
    for j = 1:s
        % find maximal index i between 0 and r, such that the first i
        % digits of the b-adic expansion of z_j and x_n,j are identical (O(r))
        i(j) = b_adic(z(j),X(j,n),b,r);
    end
    % add the values to ST_r
    ST_r = ST_r + Y(n) * algorithm_2_efficient(i,r);
end

end