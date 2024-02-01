function [Y] = Zhou_fct(X)
% evaluates the Zhou function for a set of s-dimensional points
% Input: - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
% Output: - set of N corresponding answers Y (evaluated based on the 
%           Zhou function) as real numbers (1xN-matrix)

% initialize the output, necessary varibales and functions
s = size(X,1);
N = size(X,2);
Y = zeros(1,N);
Phi = @(x) (2*pi)^(-s/2)*exp(-0.5*norm(x)^2);

% evaluate the Zhou function for every input point seperately
for i = 1:N
    Y(1,i) = 10^s/2*(Phi(10*(X(:,i)-1/3))+Phi(10*(X(:,i)-2/3)));
end

end