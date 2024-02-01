% Wir testen die Funktionen aus diesem Ordner auf ihre Korrektheit. Dazu
% wählen wir zuerst einfach per Hand nachrechenbare Beispiele und
% vergelichen die Werte. Anschließend schauen wir, ob die Funktionen auch
% für unhandliche Situation durchlaufen.


% NN_stencil.m:
X = reshape(1:400,20,20); 
[sten1] = NN_stencil(X,2); % erwarteter Eintrag (1,1):11.5 (1,2):51.5 (2,1):13.5 (10,10):389.5
[sten2] = NN_stencil(X,4); % erwarteter Eintrag (1,1):32.5 (1,2):112.5 (2,1):36.5 (5,5):368.5
[sten3] = NN_stencil(X,5); % erwarteter Eintrag (1,1):43 (1,2):143 (2,1):48 (4,4):358
[sten4] = NN_stencil(X,10); % erwarteter Eintrag (1,1):95.5 (1,2):105.5 (2,1):295.5 (2,2):305.5

Y = ones(9,9); % Wir erwarten immer 1 als Durchschnitt, wenn er berechenbar ist.
[sten5] = NN_stencil(Y,2); % Nicht berechenbar, da 2 nicht 9 ganzzahlig teilt.
[sten6] = NN_stencil(Y,3);
[sten7] = NN_stencil(Y,9);


% NN_Fehler.m:
% Im Fall von handlichen Beispielen müssen wir die Gleichverteilung
% verwenden, da es sonst nicht nachrechenbar ist. Außerdem muss a und b der
% gleiche Wert sein. 
% Zunächst nur eine Schicht.
X = reshape(1:16,4,4);
Y = zeros(1,4);
P = [1 2;1 2;1 2;1 2];
W = [0 1; 1 0];
C = 4;
[err_app1] = err_app_NN(1,1,1,X,Y,P,W,C,'uni'); % erwarteter Wert: 1420
W = zeros(2,2);
[err_app2] = err_app_NN(1,1,1,X,Y,P,W,C,'uni'); % erwarteter Wert: 1476
W = ones(2,2);
Y = ones(1,4);
[err_app3] = err_app_NN(1,1,1,X,Y,P,W,C,'uni'); % erwarteter Wert: 1352
X = X';
Y = 4*ones(1,4);
[err_app4] = err_app_NN(1,1,1,X,Y,P,W,C,'uni'); % erwarteter Wert: 848
[err_app5] = err_app_NN(1,-1,-1,X,Y,P,W,C,'uni'); % erwarteter Wert: 1344

% 2 Schichten.
C = [4 2];
[err_app6] = err_app_NN(1,1,1,X,Y,P,W,C,'uni'); % erwarteter Wert: 11.928
W = zeros(2,2);
[err_app7] = err_app_NN(1,1,1,X,Y,P,W,C,'uni'); % erwarteter Wert: 12
W = ones(2,2);
Y = zeros(1,4);
[err_app8] = err_app_NN(1,1,1,X,Y,P,W,C,'uni'); % erwarteter Wert: 4.072

% Wir verwenden nun viele Schichten und zufällige Werte. Ab hier rechnen
% wir demnach nicht mehr per Hand nach.
s = 200;
X = randn(s,10^5);
Y = rand(1,s);
P = randn(s,2^10);
W = rand(2,2^10);
C = [200,100,40,20,5];
[err_app9] = err_app_NN(100,-1,1,X,Y,P,W,C,'uni'); % Gleichverteilt.
[err_app10] = err_app_NN(100,-1,1,X,Y,P,W,C,'norm'); % Normalverteilung.
[err_app11] = err_app_NN(100,-1,1,X,Y,P,W,C,'quasi'); % Ungültig.


