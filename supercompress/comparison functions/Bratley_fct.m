function [Y] = Bratley_fct(X)
% evaluates the Bratley function for a set of s-dimensional points
% Input: - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
% Output: - set of N corresponding answers Y (evaluated based on the 
%           Bratley function) as real numbers (1xN-matrix)

% initialize the output and necessary varibales
s = size(X,1);
N = size(X,2);
Y = zeros(1,N);

% evaluate the Bratley function for every input point seperately
for i = 1:N
    for j = 1:s
        Y(1,i) = Y(1,i) + (-1)^(j)*prod(X(1:j,i));
    end
end

end