% This script compares the error function and accuracy of the kmeans and 
% the weighted QMC_Voronoi. The evaluation will be done over 100 random 
% samples and meaned in the end to make the result as robust as possible.

%% initialization of the variables and errors
addpath(genpath([pwd, filesep, '\comparison functions']));
N = 3000; % number of original points
b = 2;
m = 10;
K = b^m; % number of compressed points
s = 2; % dimension

% continuous integrand
err_continuous_integrand = 0;
err_supercom_continuous_integrand = 0;
err_QMCVor_continuous_integrand = 0;

% % corner peak integrand
% err_corner_peak_integrand = 0;
% err_supercom_corner_peak_integrand = 0;
% err_QMCVor_corner_peak_integrand = 0;

% discontinuous integrand
err_discontinuous_integrand = 0;
err_supercom_discontinuous_integrand = 0;
err_QMCVor_discontinuous_integrand = 0;

% % oscillatory integrand
% err_oscillatory_integrand = 0;
% err_supercom_oscillatory_integrand = 0;
% err_QMCVor_oscillatory_integrand = 0;

% % G-Function
% err_Gfct = 0;
% err_supercom_Gfct = 0;
% err_QMCVor_Gfct = 0;
% 
% % Morokoff & Caflisch function 1
% err_MC = 0;
% err_supercom_MC = 0;
% err_QMCVor_MC = 0;
% 
% % Bratley function
% err_Bratley = 0;
% err_supercom_Bratley = 0;
% err_QMCVor_Bratley = 0;

% Zhou function
err_Zhou = 0;
err_supercom_Zhou = 0;
err_QMCVor_Zhou = 0;

% % environmental model function
% err_environmental = 0;
% err_supercom_environmental = 0;
% err_QMCVor_environmental = 0;

%% calculation loop
number = 100;
for z = 1:number
    % generation of the random samples
    X = rand(s,N);
    
    %% Evaluation of the different functions
    
    % continuous integrand
    a = 5*ones(s,1);
    u = 0.5*ones(s,1);
    Y_continuous_integrand = exp(-sum(a.*abs(X-u),1)) + normrnd(0,0.02,1,N);

    % % corner peak integrand
    % a = 5*ones(s,1);
    % Y_corner_peak_integrand = (1+sum(a.*X,1)).^(-s-1) + normrnd(0,0.02,1,N);

    % discontinuous integrand
    a = 5*ones(s,1);
    u = 0.5*ones(s,1);
    Y_discontinuous_integrand = discontinuousintegrand(X,a,u) + normrnd(0,0.02,1,N);

    % % oscillatory integrand
    % a = 5*ones(s,1);
    % u = 0.5*ones(s,1);
    % Y_oscillatory_integrand = cos(2*pi*u(1)+sum(a.*X,1)) + normrnd(0,0.02,1,N);

    % % G-Function
    % Y_Gfct = G_fct(X) + normrnd(0,0.02,1,N);
    % 
    % % Morokoff & Caflisch function 1
    % Y_MC = (1+1/s)^s*prod(X.^(1/s),1) + normrnd(0,0.02,1,N);
    % 
    % % Bratley function
    % Y_Bratley = Bratley_fct(X) + normrnd(0,0.02,1,N);

    % Zhou function
    Y_Zhou = Zhou_fct(X) + normrnd(0,0.02,1,N);

    % % environmental model function
    % if s ==2
    %     M = 10;
    %     D = 0.06;
    %     L = 1;
    %     tau = 30.1;
    %     Y_environmental = environmental_fct(X,M,D,L,tau) + normrnd(0,0.02,1,N);
    % end
    % 
    
    %% deriviation of the kmeans points
    
    % continuous integrand
    [x_continuous_integrand,y_continuous_integrand] = supercompress(X,Y_continuous_integrand,K,1);

    % % corner peak integrand
    % [x_corner_peak_integrand,y_corner_peak_integrand] = supercompress(X,Y_corner_peak_integrand,K,1);

    % discontinuous integrand
    [x_discontinuous_integrand,y_discontinuous_integrand] = supercompress(X,Y_discontinuous_integrand,K,1);

    % % oscillatory integrand
    % [x_oscillatory_integrand,y_oscillatory_integrand] = supercompress(X,Y_oscillatory_integrand,K,1);

    % % G-Function
    % [x_Gfct,y_Gfct] = supercompress(X,Y_Gfct,K,1);
    % 
    % % Morokoff & Caflisch function 1
    % [x_MC,y_MC] = supercompress(X,Y_MC,K,1);
    % 
    % % Bratley function
    % [x_Bratley,y_Bratley] = supercompress(X,Y_Bratley,K,1);

    % Zhou function
    [x_Zhou,y_Zhou] = supercompress(X,Y_Zhou,K,1);

    % % environmental model function
    % if s ==2
    %     [x_environmental,y_environmental] = supercompress(X,Y_environmental,K,1);
    % end
    
    
    %% deriviation of the QMC Method points and weights
    
    % continuous integrand
    [P_continuous_integrand,Yp_continuous_integrand] = QMCVoronoi(b,m,X,Y_continuous_integrand);

    % % corner peak integrand
    % [P_corner_peak_integrand,Yp_corner_peak_integrand] = QMCVoronoi(b,m,X,Y_corner_peak_integrand);

    % discontinuous integrand
    [P_discontinuous_integrand,Yp_discontinuous_integrand] = QMCVoronoi(b,m,X,Y_discontinuous_integrand);

    % % oscillatory integrand
    % [P_oscillatory_integrand,Yp_oscillatory_integrand] = QMCVoronoi(b,m,X,Y_oscillatory_integrand);

    % % G-Function
    % [P_Gfct,Yp_Gfct] = QMCVoronoi(b,m,X,Y_Gfct);
    % 
    % % Morokoff & Caflisch function 1
    % [P_MC,Yp_MC] = QMCVoronoi(b,m,X,Y_MC);
    % 
    % % Bratley function
    % [P_Bratley,Yp_Bratley] = QMCVoronoi(b,m,X,Y_Bratley);

    % Zhou function
    [P_Zhou,Yp_Zhou] = QMCVoronoi(b,m,X,Y_Zhou);

    % % environmental model function
    % if s ==2
    %     [P_environmental,Yp_environmental] = QMCVoronoi(b,m,X,Y_environmental);
    % end
    
    
    %% comparison of both methods based in the MSE for different functions
    
    % continuous integrand
    a = 5*ones(s,1);
    u = 0.5*ones(s,1);
    err_continuous_integrand = err_continuous_integrand + 1/N*sum((exp(-sum(a.*abs(X-u),1))-Y_continuous_integrand).^2);
    err_supercom_continuous_integrand = err_supercom_continuous_integrand + 1/K*sum((exp(-sum(a.*abs(x_continuous_integrand-u),1))-y_continuous_integrand).^2);
    err_QMCVor_continuous_integrand = err_QMCVor_continuous_integrand + 1/N*sum((exp(-sum(a.*abs(P_continuous_integrand-u),1))-Yp_continuous_integrand(1,:)).^2.*Yp_continuous_integrand(2,:));

    % % corner peak integrand
    % a = 5*ones(s,1);
    % err_corner_peak_integrand = err_corner_peak_integrand + 1/N*sum(((1+sum(a.*X,1)).^(-s-1)-Y_corner_peak_integrand).^2);
    % err_supercom_corner_peak_integrand = err_supercom_corner_peak_integrand + 1/K*sum(((1+sum(a.*x_corner_peak_integrand,1)).^(-s-1)-y_corner_peak_integrand).^2);
    % err_QMCVor_corner_peak_integrand = err_QMCVor_corner_peak_integrand + 1/N*sum(((1+sum(a.*P_corner_peak_integrand,1)).^(-s-1)-Yp_corner_peak_integrand(1,:)).^2.*Yp_corner_peak_integrand(2,:));

    % discontinuous integrand
    a = 5*ones(s,1);
    u = 0.5*ones(s,1);
    err_discontinuous_integrand = err_discontinuous_integrand + 1/N*sum((discontinuousintegrand(X,a,u)-Y_discontinuous_integrand).^2);
    err_supercom_discontinuous_integrand = err_supercom_discontinuous_integrand + 1/K*sum((discontinuousintegrand(x_discontinuous_integrand,a,u)-y_discontinuous_integrand).^2);
    err_QMCVor_discontinuous_integrand = err_QMCVor_discontinuous_integrand + 1/N*sum((discontinuousintegrand(P_discontinuous_integrand,a,u)-Yp_discontinuous_integrand(1,:)).^2.*Yp_discontinuous_integrand(2,:));

    % % oscillatory integrand
    % a = 5*ones(s,1);
    % u = 0.5*ones(s,1);
    % err_oscillatory_integrand = err_oscillatory_integrand + 1/N*sum((cos(2*pi*u(1)+sum(a.*X,1))-Y_oscillatory_integrand).^2);
    % err_supercom_oscillatory_integrand = err_supercom_oscillatory_integrand + 1/K*sum((cos(2*pi*u(1)+sum(a.*x_oscillatory_integrand,1))-y_oscillatory_integrand).^2);
    % err_QMCVor_oscillatory_integrand = err_QMCVor_oscillatory_integrand + 1/N*sum((cos(2*pi*u(1)+sum(a.*P_oscillatory_integrand,1))-Yp_oscillatory_integrand(1,:)).^2.*Yp_oscillatory_integrand(2,:));

    % % G-Function
    % err_Gfct = err_Gfct + 1/N*sum((G_fct(X)-Y_Gfct).^2);
    % err_supercom_Gfct = err_supercom_Gfct + 1/K*sum((G_fct(x_Gfct)-y_Gfct).^2);
    % err_QMCVor_Gfct = err_QMCVor_Gfct + 1/N*sum((G_fct(P_Gfct)-Yp_Gfct(1,:)).^2.*Yp_Gfct(2,:));
    % 
    % % Morokoff & Caflisch function 1
    % err_MC = err_MC + 1/N*sum(((1+1/s)^s*prod(X.^(1/s),1)-Y_MC).^2);
    % err_supercom_MC = err_supercom_MC + 1/K*sum(((1+1/s)^s*prod(x_MC.^(1/s),1)-y_MC).^2);
    % err_QMCVor_MC = err_QMCVor_MC + 1/N*sum(((1+1/s)^s*prod(P_MC.^(1/s),1)-Yp_MC(1,:)).^2.*Yp_MC(2,:));
    % 
    % % Bratley function
    % err_Bratley = err_Bratley + 1/N*sum((Bratley_fct(X)-Y_Bratley).^2);
    % err_supercom_Bratley = err_supercom_Bratley + 1/K*sum((Bratley_fct(x_Bratley)-y_Bratley).^2);
    % err_QMCVor_Bratley = err_QMCVor_Bratley + 1/N*sum((Bratley_fct(P_Bratley)-Yp_Bratley(1,:)).^2.*Yp_Bratley(2,:));

    % Zhou function
    err_Zhou = err_Zhou + 1/N*sum((Zhou_fct(X)-Y_Zhou).^2);
    err_supercom_Zhou = err_supercom_Zhou + 1/K*sum((Zhou_fct(x_Zhou)-y_Zhou).^2);
    err_QMCVor_Zhou = err_QMCVor_Zhou + 1/N*sum((Zhou_fct(P_Zhou)-Yp_Zhou(1,:)).^2.*Yp_Zhou(2,:));

    % % environmental model function
    % if s == 2
    %     M = 10;
    %     D = 0.06;
    %     L = 1;
    %     tau = 30.1;
    %     err_environmental = err_environmental + 1/N*sum((environmental_fct(X,M,D,L,tau)-Y_environmental).^2);
    %     err_supercom_environmental = err_supercom_environmental + 1/K*sum((environmental_fct(x_environmental,M,D,L,tau)-y_environmental).^2);
    %     err_QMCVor_environmental = err_QMCVor_environmental + 1/N*sum((environmental_fct(P_environmental,M,D,L,tau)-Yp_environmental(1,:)).^2.*Yp_environmental(2,:));
    % end
end

%% take the average error

% continuous integrand
err_continuous_integrand = err_continuous_integrand/number;
err_supercom_continuous_integrand = err_supercom_continuous_integrand/number;
err_QMCVor_continuous_integrand = err_QMCVor_continuous_integrand/number;

% % corner peak integrand
% err_corner_peak_integrand = err_corner_peak_integrand/number;
% err_supercom_corner_peak_integrand = err_supercom_corner_peak_integrand/number;
% err_QMCVor_corner_peak_integrand = err_QMCVor_corner_peak_integrand/number;

% discontinuous integrand
err_discontinuous_integrand = err_discontinuous_integrand/number;
err_supercom_discontinuous_integrand = err_supercom_discontinuous_integrand/number;
err_QMCVor_discontinuous_integrand = err_QMCVor_discontinuous_integrand/number;

% % oscillatory integrand
% err_oscillatory_integrand = err_oscillatory_integrand/number;
% err_supercom_oscillatory_integrand = err_supercom_oscillatory_integrand/number;
% err_QMCVor_oscillatory_integrand = err_QMCVor_oscillatory_integrand/number;

% % G-Function
% err_Gfct = err_Gfct/number;
% err_supercom_Gfct = err_supercom_Gfct/number;
% err_QMCVor_Gfct = err_QMCVor_Gfct/number;
% 
% % Morokoff & Caflisch function 1
% err_MC = err_MC/number;
% err_supercom_MC = err_supercom_MC/number;
% err_QMCVor_MC = err_QMCVor_MC/number;
% 
% % Bratley function
% err_Bratley = err_Bratley/number;
% err_supercom_Bratley = err_supercom_Bratley/number;
% err_QMCVor_Bratley = err_QMCVor_Bratley/number;

% Zhou function
err_Zhou = err_Zhou/number;
err_supercom_Zhou = err_supercom_Zhou/number;
err_QMCVor_Zhou = err_QMCVor_Zhou/number;

% % environmental model function
% err_environmental = err_environmental/number;
% err_supercom_environmental = err_supercom_environmental/number;
% err_QMCVor_environmental = err_QMCVor_environmental/number;


%% caculate the accuracy

% continuous integrand
accuracy_supercom_continuous_integrand = abs(err_continuous_integrand-err_supercom_continuous_integrand);
accuracy_QMCVor_continuous_integrand = abs(err_continuous_integrand-err_QMCVor_continuous_integrand);

% % corner peak integrand
% accuracy_supercom_corner_peak_integrand = abs(err_corner_peak_integrand-err_supercom_corner_peak_integrand);
% accuracy_QMCVor_corner_peak_integrand = abs(err_corner_peak_integrand-err_QMCVor_corner_peak_integrand);

% discontinuous integrand
accuracy_supercom_discontinuous_integrand = abs(err_discontinuous_integrand-err_supercom_discontinuous_integrand);
accuracy_QMCVor_discontinuous_integrand = abs(err_discontinuous_integrand-err_QMCVor_discontinuous_integrand);

% % oscillatory integrand
% accuracy_supercom_oscillatory_integrand = abs(err_oscillatory_integrand-err_supercom_oscillatory_integrand);
% accuracy_QMCVor_oscillatory_integrand = abs(err_oscillatory_integrand-err_QMCVor_oscillatory_integrand);

% % G-Function
% accuracy_supercom_Gfct = abs(err_Gfct-err_supercom_Gfct);
% accuracy_QMCVor_Gfct = abs(err_Gfct-err_QMCVor_Gfct);
% 
% % Morokoff & Caflisch function 1
% accuracy_supercom_MC = abs(err_MC-err_supercom_MC);
% accuracy_QMCVor_MC = abs(err_MC-err_QMCVor_MC);
% 
% % Bratley function
% accuracy_supercom_Bratley = abs(err_Bratley-err_supercom_Bratley);
% accuracy_QMCVor_Bratley = abs(err_Bratley-err_QMCVor_Bratley);

% Zhou function
accuracy_supercom_Zhou = abs(err_Zhou-err_supercom_Zhou);
accuracy_QMCVor_Zhou = abs(err_Zhou-err_QMCVor_Zhou);

% % environmental model function
% accuracy_supercom_environmental = abs(err_environmental-err_supercom_environmental);
% accuracy_QMCVor_environmental = abs(err_environmental-err_QMCVor_environmental);

err = [accuracy_supercom_continuous_integrand accuracy_supercom_discontinuous_integrand  accuracy_supercom_Zhou; ...
       accuracy_QMCVor_continuous_integrand  accuracy_QMCVor_discontinuous_integrand  accuracy_QMCVor_Zhou];
save(sprintf('err_weightQMCVor_%d_%d_%d.mat',[N s K]),'err')

% accuracy_supercom_corner_peak_integrand  accuracy_supercom_oscillatory_integrand  accuracy_supercom_Gfct  accuracy_supercom_MC  accuracy_supercom_Bratley   accuracy_supercom_environmental
% accuracy_QMCVor_corner_peak_integrand  accuracy_QMCVor_oscillatory_integrand  accuracy_QMCVor_Gfct  accuracy_QMCVor_MC  accuracy_QMCVor_Bratley accuracy_QMCVor_environmental