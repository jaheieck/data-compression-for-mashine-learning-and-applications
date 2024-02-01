% This script compares the error function and accuracy of the supercompress 
% and the QMC compression, if the function values are scaled
% differently for the discontinous integrands.

% initialization of the variables and errors
addpath(genpath([pwd, filesep, '\comparison functions']));
N = 3000; % number of original points
b = 2;
m = 9;
K = b^m; % number of compressed points
s = 2; % dimension

% discontinuous integrand
err_discontinuous_integrand = 0;
err_supercom_discontinuous_integrand = 0;
err_QMC_discontinuous_integrand = 0;

% scaled to max value 100
err_discontinuous_integrand_small_scale = 0;
err_supercom_discontinuous_integrand_small_scale = 0;
err_QMC_discontinuous_integrand_small_scale = 0;

% scaled to max value 10
err_discontinuous_integrand_medium_scale = 0;
err_supercom_discontinuous_integrand_medium_scale = 0;
err_QMC_discontinuous_integrand_medium_scale = 0;

% scaled to max value 1
err_discontinuous_integrand_big_scale = 0;
err_supercom_discontinuous_integrand_big_scale = 0;
err_QMC_discontinuous_integrand_big_scale = 0;


% calculation loop
number = 100;
for z = 1:number
    % generation of the random samples
    X = rand(s,N);
    
    % evaluation of the discontinuous integrand function
    a = 5*ones(s,1);
    u = 0.5*ones(s,1);
    Y_discontinuous_integrand = discontinuousintegrand(X,a,u) + normrnd(0,0.02,1,N); % add noise to original data
    Y_discontinuous_integrand_small_scale = Y_discontinuous_integrand/max(Y_discontinuous_integrand)*100; % scaled to max 100
    Y_discontinuous_integrand_medium_scale = Y_discontinuous_integrand/max(Y_discontinuous_integrand)*10; % scaled to max 10
    Y_discontinuous_integrand_big_scale = Y_discontinuous_integrand/max(Y_discontinuous_integrand); % scaled to max 1
    
    % deriviation of the kmeans points
    [x_discontinuous_integrand,y_discontinuous_integrand] = supercompress(X,Y_discontinuous_integrand,K,1);
    [x_discontinuous_integrand_small_scale,y_discontinuous_integrand_small_scale] = supercompress(X,Y_discontinuous_integrand_small_scale,K,1);
    [x_discontinuous_integrand_medium_scale,y_discontinuous_integrand_medium_scale] = supercompress(X,Y_discontinuous_integrand_medium_scale,K,1);
    [x_discontinuous_integrand_big_scale,y_discontinuous_integrand_big_scale] = supercompress(X,Y_discontinuous_integrand_big_scale,K,1);
     
    % deriviation of the QMC Method points and weights
    [P_discontinuous_integrand,W_discontinuous_integrand] = QMC(b,m,X,Y_discontinuous_integrand);
    [P_discontinuous_integrand_small_scale,W_discontinuous_integrand_small_scale] = QMC(b,m,X,Y_discontinuous_integrand_small_scale);
    [P_discontinuous_integrand_medium_scale,W_discontinuous_integrand_medium_scale] = QMC(b,m,X,Y_discontinuous_integrand_medium_scale);
    [P_discontinuous_integrand_big_scale,W_discontinuous_integrand_big_scale] = QMC(b,m,X,Y_discontinuous_integrand_big_scale);

 
    % comparison of both methods based in the MSE for different functions
    a = 5*ones(s,1);
    u = 0.5*ones(s,1);

    err_discontinuous_integrand = err_discontinuous_integrand + 1/N*sum((discontinuousintegrand(X,a,u)-Y_discontinuous_integrand).^2);
    err_supercom_discontinuous_integrand = err_supercom_discontinuous_integrand + 1/K*sum((discontinuousintegrand(x_discontinuous_integrand,a,u)-y_discontinuous_integrand).^2);
    err_QMC_discontinuous_integrand = err_QMC_discontinuous_integrand + sum(discontinuousintegrand(P_discontinuous_integrand,a,u).^2.*W_discontinuous_integrand(1,:)) - 2*sum(discontinuousintegrand(P_discontinuous_integrand,a,u).*W_discontinuous_integrand(2,:)) + 1/N*sum(Y_discontinuous_integrand.^2);

    err_discontinuous_integrand_small_scale = err_discontinuous_integrand_small_scale + 1/N*sum((discontinuousintegrand(X,a,u)/discontinuousintegrand(X,a,u)*100-Y_discontinuous_integrand_small_scale).^2);
    err_supercom_discontinuous_integrand_small_scale = err_supercom_discontinuous_integrand_small_scale + 1/K*sum((discontinuousintegrand(x_discontinuous_integrand_small_scale,a,u)/max(discontinuousintegrand(x_discontinuous_integrand_small_scale,a,u))*100-y_discontinuous_integrand_small_scale).^2);
    err_QMC_discontinuous_integrand_small_scale = err_QMC_discontinuous_integrand_small_scale + sum(discontinuousintegrand(P_discontinuous_integrand_small_scale,a,u)/max(discontinuousintegrand(P_discontinuous_integrand_small_scale,a,u))*100.^2.*W_discontinuous_integrand_small_scale(1,:)) - 2*sum(discontinuousintegrand(P_discontinuous_integrand_small_scale,a,u)/max(discontinuousintegrand(P_discontinuous_integrand_small_scale,a,u))*100.*W_discontinuous_integrand_small_scale(2,:)) + 1/N*sum(Y_discontinuous_integrand_small_scale.^2);

    err_discontinuous_integrand_medium_scale = err_discontinuous_integrand_medium_scale + 1/N*sum((discontinuousintegrand(X,a,u)/max(discontinuousintegrand(X,a,u))*10-Y_discontinuous_integrand_medium_scale).^2);
    err_supercom_discontinuous_integrand_medium_scale = err_supercom_discontinuous_integrand_medium_scale + 1/K*sum((discontinuousintegrand(x_discontinuous_integrand_medium_scale,a,u)/max(discontinuousintegrand(x_discontinuous_integrand_medium_scale,a,u))*10-y_discontinuous_integrand_medium_scale).^2);
    err_QMC_discontinuous_integrand_medium_scale = err_QMC_discontinuous_integrand_medium_scale + sum(discontinuousintegrand(P_discontinuous_integrand_medium_scale,a,u)/max(discontinuousintegrand(P_discontinuous_integrand_medium_scale,a,u))*10.^2.*W_discontinuous_integrand_medium_scale(1,:)) - 2*sum(discontinuousintegrand(P_discontinuous_integrand_medium_scale,a,u)/max(discontinuousintegrand(P_discontinuous_integrand_medium_scale,a,u))*10.*W_discontinuous_integrand_medium_scale(2,:)) + 1/N*sum(Y_discontinuous_integrand_medium_scale.^2);

    err_discontinuous_integrand_big_scale = err_discontinuous_integrand_big_scale + 1/N*sum((discontinuousintegrand(X,a,u)/max(discontinuousintegrand(X,a,u))-Y_discontinuous_integrand_big_scale).^2);
    err_supercom_discontinuous_integrand_big_scale = err_supercom_discontinuous_integrand_big_scale + 1/K*sum((discontinuousintegrand(x_discontinuous_integrand_big_scale,a,u)/max(discontinuousintegrand(x_discontinuous_integrand_big_scale,a,u))-y_discontinuous_integrand_big_scale).^2);
    err_QMC_discontinuous_integrand_big_scale = err_QMC_discontinuous_integrand_big_scale + sum(discontinuousintegrand(P_discontinuous_integrand_big_scale,a,u)/max(discontinuousintegrand(P_discontinuous_integrand_big_scale,a,u)).^2.*W_discontinuous_integrand_big_scale(1,:)) - 2*sum(discontinuousintegrand(P_discontinuous_integrand_big_scale,a,u)/max(discontinuousintegrand(P_discontinuous_integrand_big_scale,a,u)).*W_discontinuous_integrand_big_scale(2,:)) + 1/N*sum(Y_discontinuous_integrand_big_scale.^2);

end

%% take the average error
err_discontinuous_integrand = err_discontinuous_integrand/number;
err_supercom_discontinuous_integrand = err_supercom_discontinuous_integrand/number;
err_QMC_discontinuous_integrand = err_QMC_discontinuous_integrand/number;
err_discontinuous_integrand_small_scale = err_discontinuous_integrand_small_scale/number;
err_supercom_discontinuous_integrand_small_scale = err_supercom_discontinuous_integrand_small_scale/number;
err_QMC_discontinuous_integrand_small_scale = err_QMC_discontinuous_integrand_small_scale/number;
err_discontinuous_integrand_medium_scale = err_discontinuous_integrand_medium_scale/number;
err_supercom_discontinuous_integrand_medium_scale = err_supercom_discontinuous_integrand_medium_scale/number;
err_QMC_discontinuous_integrand_medium_scale = err_QMC_discontinuous_integrand_medium_scale/number;
err_discontinuous_integrand_big_scale = err_discontinuous_integrand_big_scale/number;
err_supercom_discontinuous_integrand_big_scale = err_supercom_discontinuous_integrand_big_scale/number;
err_QMC_discontinuous_integrand_big_scale = err_QMC_discontinuous_integrand_big_scale/number;

% caculate the accuracy
accuracy_supercom_discontinuous_integrand = abs(err_discontinuous_integrand-err_supercom_discontinuous_integrand);
accuracy_QMC_discontinuous_integrand = abs(err_discontinuous_integrand-err_QMC_discontinuous_integrand);
accuracy_supercom_discontinuous_integrand_small_scale = abs(err_discontinuous_integrand_small_scale-err_supercom_discontinuous_integrand_small_scale);
accuracy_QMC_discontinuous_integrand_small_scale = abs(err_discontinuous_integrand_small_scale-err_QMC_discontinuous_integrand_small_scale);
accuracy_supercom_discontinuous_integrand_medium_scale = abs(err_discontinuous_integrand_medium_scale-err_supercom_discontinuous_integrand_medium_scale);
accuracy_QMC_discontinuous_integrand_medium_scale = abs(err_discontinuous_integrand_medium_scale-err_QMC_discontinuous_integrand_medium_scale);
accuracy_supercom_discontinuous_integrand_big_scale = abs(err_discontinuous_integrand_big_scale-err_supercom_discontinuous_integrand_big_scale);
accuracy_QMC_discontinuous_integrand_big_scale = abs(err_discontinuous_integrand_big_scale-err_QMC_discontinuous_integrand_big_scale);

% save the data
err = [accuracy_supercom_discontinuous_integrand  accuracy_supercom_discontinuous_integrand_small_scale  accuracy_supercom_discontinuous_integrand_medium_scale  accuracy_supercom_discontinuous_integrand_big_scale;
       accuracy_QMC_discontinuous_integrand accuracy_QMC_discontinuous_integrand_small_scale accuracy_QMC_discontinuous_integrand_medium_scale accuracy_QMC_discontinuous_integrand_big_scale];
save(sprintf('err_scale_%d_%d_%d.mat',[N s K]),'err')
