function [P,Yp] = QMCVoronoi(b,m,X,Y)
% calculation of the set of compressed data points P and its corresponding
% answers Yp od the QMC Voronoi compression method
% Input: - set of N input data points X as column vectors in [0,1)^s
%          (sxN-matrix)
%        - set of N corresponding answers Y as real numbers
%          (1xN-matrix)
%        - prime number b for the b-adic expansion
%        - natural number m, determining the number of points in P
% Output: - compressed data set P (sxN-matrix)
%         - corresponding answers Yp (2xL-matrix), where the first row are
%           the answers and the second row the number of compressed points


% load the necessary data paths and variables
addpath(genpath([fileparts(pwd), filesep, '\QMCweights']));
addpath(genpath([fileparts(pwd), filesep, '\QMCpoints']));
N = size(X,2);
s = size(X,1);
L = b^m;
load 'DIGSEQ\sobolmats\Sobol_Cs.col'
digitalseq_b2g('init0', Sobol_Cs)
P = digitalseq_b2g(s,L); % generate QMC points

% Calculate the average responses
Yp = zeros(2,L);
for i = 1:N
    [~,cellIndex] = min(pdist2(P',X(:,i)','minkowski',2));
    Yp(1,cellIndex) = Yp(1,cellIndex) + Y(i);
    Yp(2,cellIndex) = Yp(2,cellIndex) + 1;
end
P = P(:,Yp(2,:)~=0);
Yp = Yp(:,Yp(2,:)~=0); % delete empty Voronoi regions
Yp(1,Yp(2,:)~=0) = Yp(1,Yp(2,:)~=0)./Yp(2,Yp(2,:)~=0);


end