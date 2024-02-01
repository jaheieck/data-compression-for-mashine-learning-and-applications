function [W] = weight(z,X,Y,b,m,nu)
% calculation of the weights W_X,P,nu,l and W_X,Y,P,nu,l with the algorithms
% and with the formula from section 2.1.3 (O(nus^2N))
% Input: - column vector z in [0,1)^s, which is part of P
%        - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
%        - set of N corresponding answers Y as real numbers
%          (1xN-matrix), with Y = ones(1,N) for the calculation of W_X,P,nu,l
%        - prime number b for the b-adic expansion
%        - natural number m, determing the number of points L=b^m
%        - natural number nu, determining hof fine the partition is
% Output: - weight W_X,P,nu,l or W_X,Y,P,nu,l for the deriviation of the
%           error


% definiton and initialization of the variables
s = length(z);
N = length(Y);
W = 0;

% add the summand of the sum over q of the formula in section 2.1.3 step
% by step. Use Algorithm (1) for that 
for q = 0:min(s-1,nu)
    W = W + (-1)^q * nchoosek(s-1,q) * 1/b^q * algorithm_1(z,X,Y,b,nu-q);
end

% multiply with the missing factor
W = b^(nu-m)/N * W;

end