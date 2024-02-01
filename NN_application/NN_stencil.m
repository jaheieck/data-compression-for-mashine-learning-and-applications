function [sten] = NN_stencil(X,len)
% reduce the high dimensional data by taking the lenxlen sub-matrix of each
% X(:,:,1,n) it by its average value
% Input: - number of N data points X in [0,1]^s
%          ((sqrt(s)xsqrt(s)x1xN)-Matrix)
%        - len determines, how big th submatrix is
% Output: - compressed data
% Caution: we assume that the first two dimensions of X are divisible by
% len! 

% initialize the relevant parameters
N = size(X,4);
big = size(X,1);
komp = big/len; % what sized is used to compress the data?
if floor(komp) ~= komp
    fprintf('Die Eingabe von len ist nicht zulässig.\n')
    sten = [];
    return
else
sten = zeros(komp,komp,1,N);
end

% substitute submatrix by its average value
for n = 1:N
    Y = reshape(X(:,:,1,n), len, komp, len, komp);
    Y = sum(sum(Y,1),3)/len^2;
    sten(:,:,1,n) = reshape(Y, komp, komp);
end

end