function [W] = algorithm_5(P,X,Y,b,m,nu)
% Calculation of all weights W_X,P,nu,l and W_X,Y,P,nu,l with a faster
% method (O(nus^2N(nu+s-1)^(s-1)b^(m-nu+s-1))) 
% Input: - compressed point set P (sxb^m-Matrix), where the l-th
%          column vector is z_l-1 in [0,1)^s
%        - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
%        - set of N corresponding answers Y as real numbers
%          (1xN-matrix)
%        - prime number b as the base of the expansion
%        - natural number m, which determines the number of points L=b^m
%        - natural number nu, which determines how fine the partition is
% Output: - weight-matrix W (2xb^m-Matrix), where the first row is
%           W_X,P,nu,l and the second one is W_X,Y,P,nu,l
% Caution: P has to be a digital net!


% initialize variables
s = size(P,1);
L = b^m;
N = length(Y);
W = zeros(2,L);

% add step by step all summands for q  
for q = 0:min(s-1,nu)
    % save the values from which we can skip z_l
    bound = nchoosek(nu-q+s-1,s-1)*b^(m-nu+q);
    R = zeros(1,N);
    for n = 1:N
        l = 1;
        % as long as the bound is not reached and P still has points, which
        % are not taken into account yet, keep updating weights
        while l <= L && R(n) < bound
            i = zeros(1,s);
            for j = 1:s
                % find maximal index i between 0 and r, such that te first
                % i digits of the b-adic expansion of z_l,j and x_n,j are
                % identical (O(nu))
                i(j) = b_adic(P(j,l),X(j,n),b,nu-q);
            end
            % calculate N_v-q_i (O(snu))
            [N_q_i] = algorithm_2_efficient(i,nu-q);
            if N_q_i > 0
                % update the weights with the already known value
                % N_q_i. Take care of the factor in front of N_q_i
                W(1,l) = W(1,l) + (-1)^q * nchoosek(s-1,q) * 1/b^q * N_q_i;
                W(2,l) = W(2,l) + (-1)^q * nchoosek(s-1,q) * 1/b^q * Y(n) * N_q_i;
                R(1,n) = R(1,n) + N_q_i;
            end
            l = l+1;
        end
    end
end

% multiply missing factor
W = b^(nu-m)/N * W;

end