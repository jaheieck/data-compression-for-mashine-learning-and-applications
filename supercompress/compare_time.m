% This script compares the running time of QMC, supercompress and QMCVoronoi.
% The evaluation will be done over 100 random samples and 
% meaned in the end to make the result as robust as possible.

%% initialization of the variables and errors
addpath(genpath([pwd, filesep, '\comparison functions']));
N = 10000; % number of original points
b = 2;
m = 10;
K = b^m; % number of compressed points
s = 3; % dimension
number = 20; % number of repetitions

time_supercom_discontinuous_integrand = zeros(1,number); % running time supercompress
time_QMC_discontinuous_integrand = zeros(1,number); % running time QMC
time_QMCVor_discontinuous_integrand = zeros(1,number); % running time QMCVoronoi


%% calculation loop
for z = 1:number
    % generation of the random samples
    X = rand(s,N);
    
    % evaluation of the discontinuous integrand function
    a = 5*ones(s,1);
    u = 0.5*ones(s,1);
    Y_discontinuous_integrand = discontinuousintegrand(X,a,u);

    %% deriviation of the compressed sets and extraction of time

    % deriviation of the supercompress points
    tic
    [x_discontinuous_integrand,y_discontinuous_integrand] = supercompress(X,Y_discontinuous_integrand,K,0);
    time_supercom_discontinuous_integrand(1,number) = toc;
     
    tic
    % deriviation of the QMC Method points and weights
    [P_discontinuous_integrand,W_discontinuous_integrand] = QMC(b,m,X,Y_discontinuous_integrand);
    time_QMC_discontinuous_integrand(1,number) = toc;

    tic
    % deriviation of the QMC Method points and weights
    [PQ_discontinuous_integrand,Yp_discontinuous_integrand] = QMCVoronoi(b,m,X,Y_discontinuous_integrand);
    time_QMCVor_discontinuous_integrand(1,number) = toc;

end

%% save the average running time

time = [mean(time_supercom_discontinuous_integrand)  mean(time_QMC_discontinuous_integrand)  mean(time_QMCVor_discontinuous_integrand)];
save(sprintf('time_%d_%d.mat',[s K]),'time')