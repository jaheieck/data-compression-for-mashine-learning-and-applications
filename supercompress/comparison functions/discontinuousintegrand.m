function [Y] = discontinuousintegrand(X,a,u)
% evaluates functions of the discontinuous integrand family for a set of 
% s-dimensional points
% Input: - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
%        - column vector u in [0,1)^s
%        - column vector a with real valued entries
% Output: - set of N corresponding answers Y (evaluated based on the 
%           discontinuous integrand function) as real numbers (1xN-matrix)

% initialize the output and necessary varibales
N = size(X,2);
Y = zeros(1,N);

% evaluate the discontinuous integrand function for every input point seperately
for i = 1:N
    if X(1,i) <= u(1) || X(2,i) <= u(2)
        Y(1,i) = exp(sum(a.*X(:,i)));
    end
end

end