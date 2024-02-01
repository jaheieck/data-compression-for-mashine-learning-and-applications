function [Y] = G_fct(X)
% evaluates the G-function for a set of s-dimensional points
% Input: - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
% Output: - set of N corresponding answers Y (evaluated based on the 
%           G-function) as real numbers (1xN-matrix)

% initialize the output and necessary varibales
s = size(X,1);
a = zeros(s,1);

% evaluate the G-function for every input point seperately
for i = 1:s
    a(i) = (i-1)/2;
end
Y = prod(abs(4*X-2)+a./(1+a),1);

end