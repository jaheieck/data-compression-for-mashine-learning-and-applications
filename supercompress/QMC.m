function [P,W] = QMC(b,m,X,Y)
% calculation of the set of compressed data points P and its weights W for
% the classical QMC compression method from Feischl and Dick
% Input: - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
%        - set of N corresponding answers Y as real numbers
%          (1xN-matrix)
%        - prime number b for the b-adic expansion
%        - natural number m, determining the number of points in P
% Output: - compressed data set P (sxN-matrix)
%         - corresponding weights W (2xL-matrix), where the first row are
%           all weights W_X,P,nu,l and the second W_X,Y,P,nu,l


% load the necessary data paths and variables
addpath(genpath([fileparts(pwd), filesep, '\QMCweights']));
addpath(genpath([fileparts(pwd), filesep, '\QMCpoints']));
N = size(X,2);
s = size(X,1);
L = b^m;
alpha = 1;
nu = floor(m*alpha/(1+alpha));
load 'DIGSEQ\nxmats\nx_b2_m30_s16_Cs.col'
digitalseq_b2g('init0', nx_b2_m30_s16_Cs)
P = digitalseq_b2g(s,L); % generate QMC points

% calculate weights
W = zeros(2,L);
for l = 1:L 
   W(1,l) = weight(P(:,l),X,ones(1,N),b,m,nu);
   W(2,l) = weight(P(:,l),X,Y,b,m,nu);
end


end