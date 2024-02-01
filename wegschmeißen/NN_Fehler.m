% Wir berechnen nun wie bei Dick und Feischl den Approximationsfehler für
% Neuronale Netze. Vorsicht: Die Ausführung dieses Codes dauert
% länger (bei mir ca. 290 min.)!
% Der Fehler wird für verschiedene Größen L und Sobol-Matrizen berechnet, da 
% nur sie sich für so hohe Dimensionen eignen. X und Y werden aus dem Datenset
% MNIST von https://yann.lecun.com/exdb/mnist/ übernommen. Wir wählen die 
% einzelnen Komponenten von theta standardnormalverteilt. Der Fehler wird 
% über 100 für verschiedene Wahlen von theta berechnet. Danach werden alle
% Fehler visualisiert. Da der Aufwand für die verschiedenen Fälle sehr hoch
% ist, verwenden wir die effizientere Variante für die Berechnug der
% Gewichte aus Schrittweise Berechnung der Gewichte.
% Achtung: Deep Learning Toolbox muss gedownloadet werden!

% Lade die notwendigen Ordner, die Funktionen beinhalten, die wir brauchen.
addpath(genpath([fileparts(pwd), filesep, '\Standardverfahren Berechnung der Gewichte']));
addpath(genpath([fileparts(pwd), filesep, '\Schrittweise Berechnung der Gewichte']));
addpath(genpath([fileparts(pwd), filesep, '\Punkteberechnung']));
addpath(genpath([pwd, filesep, '\MNIST']));

% Daten laden von https://yann.lecun.com/exdb/mnist/
XTrain = processImagesMNIST('train-images-idx3-ubyte.gz');
YTrain = processLabelsMNIST('train-labels-idx1-ubyte.gz');

% Komprimiere die Daten. Wir nehmen nur 10000 Graubilder statt 60000, um die
% Laufzeit kleiner zu halten. Außerdem stellen wir 4x4 Pixel durch einen
% Grauwert dar.
XTrain = XTrain(:,:,:,1:10^4);
YTrain_komp = YTrain(1,1:10^4);
[XTrain_komp] = NN_stencil(XTrain,4);

% Überführe YTrain_komp in einen Vektor mit Zahlen.
YTrain_komp_help = cellstr(YTrain_komp);
YTrain_komp_help = str2double(YTrain_komp_help);

% Extrahiere die Parameter.
N = size(XTrain_komp,4);
s = size(XTrain_komp,1)^2;

% Schreibe die Daten um, sodass sie die passenden Form für unsere
% Algorithmen haben.
XTrain_komp = reshape(XTrain_komp,[],N);
% Rücktransformation:
% XTrain_komp = reshape(XTrain_komp,size(XTrain_komp,1),size(XTrain_komp,1),1,N);

% Definiere Parameter.
b = 2;
nu = 2; 

% Berechne den Fehler für ein tiefes und ein untiefes Neuronales Netz, wenn
% die Punktmenge die Ordnung 1 hat.
err_app_shallow_1 = zeros(4,100);
err_app_deep_1 = zeros(4,100);

tic
for m = 10:13
    L = b^m;
    % Berechne das (t,m,s)-Netz P mit Hilfe der Funktionen aus Punkteberechnung.
    load 'DIGSEQ\sobolmats\sobol_Cs.col'
    digitalseq_b2g('init0', sobol_Cs)
    P = digitalseq_b2g(s,L);
     
    % Berechne die Gewichte für P mit Hilfe der Funktionen aus Schrittweise 
    % Berechnung der Gewichte.
    [W] = algorithm_5(P,XTrain_komp,YTrain_komp_help,b,m,nu);

    % Berechne den Fehler.
    [err_app_shallow_1(m-9,:)] = err_app_NN(100,-1,1,XTrain_komp,YTrain_komp_help,P,W,s,'uni');
    [err_app_deep_1(m-9,:)] = err_app_NN(100,-1,1,XTrain_komp,YTrain_komp_help,P,W,[s,30,12,5],'uni');
end


% Wiederhole das gleiche für Punktmengen der Ordnung 2.
err_app_shallow_2 = zeros(4,100);
err_app_deep_2 = zeros(4,100);

for m = 10:13
    L = b^m;
    % Berechne das (t_alpha,m,s)-Netz P mit Hilfe der Funktionen aus Punkteberechnung.
    load 'DIGSEQ\sobolmats\sobol_alpha2_Bs64.col'
    digitalseq_b2g('init0', sobol_alpha2_Bs64)
    P = digitalseq_b2g(s,L);

    % Berechne die Gewichte für P mit Hilfe der Funktionen aus Schrittweise 
    % Berechnung der Gewichte.
    [W] = algorithm_5(P,XTrain_komp,YTrain_komp_help,b,m,nu);

    % Berechne den Fehler.
    [err_app_shallow_2(m-9,:)] = err_app_NN(100,-1,1,XTrain_komp,YTrain_komp_help,P,W,s,'uni');
    [err_app_deep_2(m-9,:)] = err_app_NN(100,-1,1,XTrain_komp,YTrain_komp_help,P,W,[s,30,12,5],'uni');
end
time = toc;

% Speicher die Fehler für die Visualisierung.
save('errapp_NN.mat','err_app_shallow_1','err_app_shallow_2',...
     'err_app_deep_1','err_app_deep_2');

% Berechne den Durchschnittlichen Approximationsfehler und Fehler.
avg_mistake_shallow_1 = mean(err_app_shallow_1,2);
avg_mistake_shallow_2 = mean(err_app_shallow_2,2);
avg_mistake_deep_1 = mean(err_app_deep_1,2);
avg_mistake_deep_2 = mean(err_app_deep_2,2);
