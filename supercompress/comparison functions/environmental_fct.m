function [Y] = environmental_fct(X,M,D,L,tau)
% evaluates the environmental function for a set of 
% s-dimensional points
% Input: - set of N input data points X as column vectors in [0,1)^2
%          (2xN-matrix)
%        - real value M representing the mass of pollutant
%        - real value D representing the diffusion rate
%        - real value L representing the location of the second spill
%        - real value tau representing the time of the second spill
% Output: - set of N corresponding answers Y (evaluated based on the 
%           environmental function) as real numbers (1xN-matrix)

% initialize the output and necessary varibales
N = size(X,2);
Y = zeros(1,N);

% evaluate the environmental function for every input point seperately
for j = 1:N
    si = X(1,j);
    tj = X(2,j);
    if tj == 0
        Y(1,j) = 0;
    else
        term1a = M / sqrt(4*pi*D*tj);
        term1b = exp(-si^2 / (4*D*tj));
        term1 = term1a * term1b;
    
        term2 = 0;
        if (tau < tj)
            term2a = M / sqrt(4*pi*D*(tj-tau));
            term2b = exp(-(si-L)^2 / (4*D*(tj-tau)));
            term2 = term2a * term2b;
        end
    
        C = term1 + term2;
        Y(1,j) = sqrt(4*pi) * C;
    end
end

end