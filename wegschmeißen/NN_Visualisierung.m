% Wir Visualsisieren die Daten der Fehlerberechnung des NN.

% Lade zunächst die Daten.
addpath(genpath([pwd, filesep, '\Ergebnisse']));
addpath(genpath([fileparts(pwd), filesep, '\Fehlerberechnung Neuronales Netz']));
errapp_NN_shallow_alpha1 = matfile('errapp_NN_7x7_uni.mat').err_app_shallow_1;
errapp_NN_shallow_alpha2 = matfile('errapp_NN_7x7_uni.mat').err_app_shallow_2;
errapp_NN_deep_alpha1 = matfile('errapp_NN_7x7_uni.mat').err_app_deep_1;
errapp_NN_deep_alpha2 = matfile('errapp_NN_7x7_uni.mat').err_app_deep_2;

% Extrahiere die relevanten Daten für ein Balkendiagramm, sodass für
% verschiedene Wahlen von m die Fehler evaluiert werden können.
value_shallow_1 = 11;
X_shallow_1 = 1:value_shallow_1;
Y_shallow_1_1 = zeros(1,value_shallow_1);
Y_shallow_1_2 = zeros(1,value_shallow_1);
Y_shallow_1_3 = zeros(1,value_shallow_1);
Y_shallow_1_4 = zeros(1,value_shallow_1);
for i = 0:value_shallow_1-1
    Y_shallow_1_1(1,i+1) = sum(i <= errapp_NN_shallow_alpha1(1,:) & errapp_NN_shallow_alpha1(1,:) < i+1);
    Y_shallow_1_2(1,i+1) = sum(i <= errapp_NN_shallow_alpha1(2,:) & errapp_NN_shallow_alpha1(2,:) < i+1);
    Y_shallow_1_3(1,i+1) = sum(i <= errapp_NN_shallow_alpha1(3,:) & errapp_NN_shallow_alpha1(3,:) < i+1);
    Y_shallow_1_4(1,i+1) = sum(i <= errapp_NN_shallow_alpha1(4,:) & errapp_NN_shallow_alpha1(4,:) < i+1);
end
Y_shallow_1_1(1,value_shallow_1) = sum(errapp_NN_shallow_alpha1(1,:) >= value_shallow_1);
Y_shallow_1_2(1,value_shallow_1) = sum(errapp_NN_shallow_alpha1(2,:) >= value_shallow_1);
Y_shallow_1_3(1,value_shallow_1) = sum(errapp_NN_shallow_alpha1(3,:) >= value_shallow_1);
Y_shallow_1_4(1,value_shallow_1) = sum(errapp_NN_shallow_alpha1(4,:) >= value_shallow_1);

% Wiederhole für das untiefe Netz mit Punktmenge der Ordnung 2
value_shallow_2 = 11;
X_shallow_2 = 1:value_shallow_2;
Y_shallow_2_1 = zeros(1,value_shallow_2);
Y_shallow_2_2 = zeros(1,value_shallow_2);
Y_shallow_2_3 = zeros(1,value_shallow_2);
Y_shallow_2_4 = zeros(1,value_shallow_2);
for i = 0:value_shallow_2-1
    Y_shallow_2_1(1,i+1) = sum(i <= errapp_NN_shallow_alpha2(1,:) & errapp_NN_shallow_alpha2(1,:) < i+1);
    Y_shallow_2_2(1,i+1) = sum(i <= errapp_NN_shallow_alpha2(2,:) & errapp_NN_shallow_alpha2(2,:) < i+1);
    Y_shallow_2_3(1,i+1) = sum(i <= errapp_NN_shallow_alpha2(3,:) & errapp_NN_shallow_alpha2(3,:) < i+1);
    Y_shallow_2_4(1,i+1) = sum(i <= errapp_NN_shallow_alpha2(4,:) & errapp_NN_shallow_alpha2(4,:) < i+1);
end
Y_shallow_2_1(1,value_shallow_2) = sum(errapp_NN_shallow_alpha2(1,:) >= value_shallow_2);
Y_shallow_2_2(1,value_shallow_2) = sum(errapp_NN_shallow_alpha2(2,:) >= value_shallow_2);
Y_shallow_2_3(1,value_shallow_2) = sum(errapp_NN_shallow_alpha2(3,:) >= value_shallow_2);
Y_shallow_2_4(1,value_shallow_2) = sum(errapp_NN_shallow_alpha2(4,:) >= value_shallow_2);

% Wiederhole für das tiefe Netz mit Punktmenge der Ordnung 1
value_deep_1 = 7;
X_deep_1 = 0.2:0.2:1.4;
Y_deep_1_1 = zeros(1,value_deep_1);
Y_deep_1_2 = zeros(1,value_deep_1);
Y_deep_1_3 = zeros(1,value_deep_1);
Y_deep_1_4 = zeros(1,value_deep_1);
for i = 0:0.2:1
    Y_deep_1_1(1,round(i*5+1)) = sum(i <= errapp_NN_deep_alpha1(1,:) & errapp_NN_deep_alpha1(1,:) < i+0.2);
    Y_deep_1_2(1,round(i*5+1)) = sum(i <= errapp_NN_deep_alpha1(2,:) & errapp_NN_deep_alpha1(2,:) < i+0.2);
    Y_deep_1_3(1,round(i*5+1)) = sum(i <= errapp_NN_deep_alpha1(3,:) & errapp_NN_deep_alpha1(3,:) < i+0.2);
    Y_deep_1_4(1,round(i*5+1)) = sum(i <= errapp_NN_deep_alpha1(4,:) & errapp_NN_deep_alpha1(4,:) < i+0.2);
end
Y_deep_1_1(1,value_deep_1) = sum(errapp_NN_deep_alpha1(1,:) >= 1.2);
Y_deep_1_2(1,value_deep_1) = sum(errapp_NN_deep_alpha1(2,:) >= 1.2);
Y_deep_1_3(1,value_deep_1) = sum(errapp_NN_deep_alpha1(3,:) >= 1.2);
Y_deep_1_4(1,value_deep_1) = sum(errapp_NN_deep_alpha1(4,:) >= 1.2);

% Wiederhole für das tiefe Netz mit Punktmenge der Ordnung 2
value_deep_2 = 7;
X_deep_2 = 0.2:0.2:1.4;
Y_deep_2_1 = zeros(1,value_deep_2);
Y_deep_2_2 = zeros(1,value_deep_2);
Y_deep_2_3 = zeros(1,value_deep_2);
Y_deep_2_4 = zeros(1,value_deep_2);
for i = 0:0.2:1
    Y_deep_2_1(1,round(i*5+1)) = sum(i <= errapp_NN_deep_alpha2(1,:) & errapp_NN_deep_alpha2(1,:) < i+0.2);
    Y_deep_2_2(1,round(i*5+1)) = sum(i <= errapp_NN_deep_alpha2(2,:) & errapp_NN_deep_alpha2(2,:) < i+0.2);
    Y_deep_2_3(1,round(i*5+1)) = sum(i <= errapp_NN_deep_alpha2(3,:) & errapp_NN_deep_alpha2(3,:) < i+0.2);
    Y_deep_2_4(1,round(i*5+1)) = sum(i <= errapp_NN_deep_alpha2(4,:) & errapp_NN_deep_alpha2(4,:) < i+0.2);
end
Y_deep_2_1(1,value_deep_1) = sum(errapp_NN_deep_alpha2(1,:) >= 1.2);
Y_deep_2_2(1,value_deep_1) = sum(errapp_NN_deep_alpha2(2,:) >= 1.2);
Y_deep_2_3(1,value_deep_1) = sum(errapp_NN_deep_alpha2(3,:) >= 1.2);
Y_deep_2_4(1,value_deep_1) = sum(errapp_NN_deep_alpha2(4,:) >= 1.2);


% Plotte alle Balkendiagramme.
NN_Fehler_shallow = subplot(2,1,1);
bar(X_shallow_1,[Y_shallow_1_1;Y_shallow_1_2;Y_shallow_1_3;Y_shallow_1_4]); 
legend('$m=10, \nu=2$','$m=11, \nu=2$','$m=12, \nu=2$','$m=13, \nu=2$','Interpreter','latex','Location','north')
title('Untiefes Neuronales Netz mit $\alpha=1$','Interpreter','latex')
xlabel('Fehler $|err-app|$','Interpreter','latex')
ylabel('Häufigkeit')
subplot(2,1,2);
bar(X_shallow_2,[Y_shallow_2_1;Y_shallow_2_2;Y_shallow_2_3;Y_shallow_2_4]); 
legend('$m=10, \nu=2$','$m=11, \nu=2$','$m=12, \nu=2$','$m=13, \nu=2$','Interpreter','latex','Location','north')
title('Untiefes Neuronales Netz mit $\alpha=2$','Interpreter','latex')
xlabel('Fehler $|err-app|$','Interpreter','latex')
ylabel('Häufigkeit')
% Speichere die Grafik ab.
% saveas(NN_Fehler_shallow,'NN_Fehler_shallow.png')
% saveas(NN_Fehler_shallow,'NN_Fehler_shallow','epsc')

NN_Fehler_deep = subplot(2,1,1);
bar(X_deep_1,[Y_deep_1_1;Y_deep_1_2;Y_deep_1_3;Y_deep_1_4]); 
legend('$m=10, \nu=2$','$m=11, \nu=2$','$m=12, \nu=2$','$m=13, \nu=2$','Interpreter','latex','Location','north')
title('Tiefes Neuronales Netz mit $\alpha=1$','Interpreter','latex')
xlabel('Fehler $|err-app|$','Interpreter','latex')
ylabel('Häufigkeit')
subplot(2,1,2);
bar(X_deep_2,[Y_deep_2_1;Y_deep_2_2;Y_deep_2_3;Y_deep_2_4]); 
legend('$m=10, \nu=2$','$m=11, \nu=2$','$m=12, \nu=2$','$m=13, \nu=2$','Interpreter','latex','Location','north')
title('Tiefes Neuronales Netz mit $\alpha=2$','Interpreter','latex')
xlabel('Fehler $|err-app|$','Interpreter','latex')
ylabel('Häufigkeit')
% Speichere die Grafik ab.
% saveas(NN_Fehler_deep,'NN_Fehler_deep.png')
% saveas(NN_Fehler_deep,'NN_Fehler_deep','epsc')


% Betrachte die Wohlverteiltheit des Datensatzes, indem wir die Datenpunkte 
% auf das 3-dimensionale projizieren. Vergleiche die verschiedenen
% Komprimierungen des Datensatzes.
XTrain = processImagesMNIST('train-images-idx3-ubyte.gz');
XTrain = XTrain(:,:,:,1:10^4);
[XTrain_14] = NN_stencil(XTrain,2);
[XTrain_7] = NN_stencil(XTrain,4);
XTrain = reshape(XTrain,[],28,28);
XTrain_14 = reshape(XTrain_14,[],14,14);
XTrain_7 = reshape(XTrain_7,[],7,7);
XTrain_projected = subplot(3,2,1);
sgtitle('Punkteverteilung von MNIST')
scatter3(XTrain(:,1,1),XTrain(:,2,1),XTrain(:,1,2),1,'MarkerFaceAlpha',0)
title('Größe:(28x28), Punkte:(11,21,12)')
subplot(3,2,2);
scatter3(XTrain(:,4,2),XTrain(:,7,5),XTrain(:,2,2),1,'MarkerFaceAlpha',0)
title('Größe:(28x28), Punkte:(42,75,22)')
subplot(3,2,3);
scatter3(XTrain_14(:,1,1),XTrain_14(:,2,1),XTrain_14(:,1,2),1,'MarkerFaceAlpha',0)
title('Größe:(14x14), Punkte:(11,21,12)')
subplot(3,2,4);
scatter3(XTrain_14(:,4,2),XTrain_14(:,7,5),XTrain_14(:,2,2),1,'MarkerFaceAlpha',0)
title('Größe:(14x14), Punkte:(42,75,22)')
subplot(3,2,5);
scatter3(XTrain_7(:,1,1),XTrain_7(:,2,1),XTrain_7(:,1,2),1,'MarkerFaceAlpha',0)
title('Größe:(7x7), Punkte:(11,21,12)')
subplot(3,2,6);
scatter3(XTrain_7(:,4,2),XTrain_7(:,7,5),XTrain_7(:,2,2),1,'MarkerFaceAlpha',0)
title('Größe:(7x7), Punkte:(42,75,22)')
% Speichere die Grafik ab.
% saveas(XTrain_projected,'XTrain_projected.png')
% saveas(XTrain_projected,'XTrain_projected','epsc')






