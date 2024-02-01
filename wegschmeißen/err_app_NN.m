function [err_app,err_total,app_total] = err_app_NN(T,a,b,X,Y,P,W,C,type)
% Berechnung der Differenz zwischen Fehler err und approximativen Fehlers 
% app über die Formel (1) und (2) aus der Masterarbeit mit
% verschiedene Wahlen von theta (normal oder gleichverteilt auf [a,b]).
% Input: - Anzahl T der zufällig generierten theta 
%        - minimaler Wert a, den theta in jeder Komponente annehmen darf
%        - maximaler Wert b, den theta in jeder Komponente annehmen darf
%        - Menge der N Datenpunkte X gegeben als Spaltenvektoren in [0,1]^s
%          (sxN-Matrix)
%        - Menge der N korrepsondierenden Antworten Y als reelle Zahlen
%          (1xN-Matrix)
%        - Menge der L Punkte P gegeben als Spaltenvektoren in [0,1)^s
%          (sxL-Matrix)
%        - Gewichte W der L Punkte P als reelle Zahlen (2xL-Matrix), wobei
%          die erste Zeile die Gewichte unabhängig von Y beschreibt
%        - (1xAnzahl_Schichten-Matrix) C, die die Dimensionen der einzelnen
%          Schichten des NN angibt (ohne letzte Schicht, die immer 1 ist)
%        - type (string aus {uni,norm}) gibt an, gemäß welcher Verteilung
%          theta generiert wird
% Output: - err_app, absolute Differenz zwischen dem Fehlerwert err und dem
%           approximativer Fehlerwert app 
%         - optional: Fehlerwert err err_total 
%         - optional: Fehlerwert app app_total 
% Achtung: Die Summe über y_n^2 wird nicht beachtet! Sie ist für die
% Differenz err-app nicht relevant!

% Initialisierung der Parameter für app und err.
L = size(P,2);
N = length(Y);
num = size(C,2); % Anzahl der Schichten des NN
err_app = zeros(1,T);
err_total = zeros(1,T);
app_total = zeros(1,T);

% Stelle sicher, dass type zulässig ist.
if strcmp(type,'uni') == 0 && strcmp(type,'norm') == 0
    fprintf('Die Eingabe von type ist unzulässig.\n')
    err_app = [];
    return
end

for t = 1:T
    % Generiere theta.
    for lay = 1:num-1
        if strcmp(type,'uni') == 1
            % Generiere theta stückweise (Q für Q) gleichverteilt auf [a,b].
            eval(sprintf('D%d = (b-a).*rand(C(lay+1),C(lay)) + a;', lay));
        else
            % Generiere theta stückweise (Q für Q) normalverteilt.
            eval(sprintf('D%d = randn(C(lay+1),C(lay));', lay));
        end
    end
    % Generiere für die letzte Schicht Q.        
    if strcmp(type,'uni') == 1
        eval(sprintf('D%d = (b-a).*rand(1,C(num)) + a;', num));
    else
        eval(sprintf('D%d = randn(1,C(num));', num));
    end
    % Stückweise Berechnung von app.
    app = 0;
    for l = 1:L
        % Initialisiere den Funktionswert f für das NN.
        f = P(:,l);
        for lay = 1:num-1
            % Werte die Funktion f des NN stückweise aus.
            f = sigmoid([dlarray(eval(sprintf('D%d',lay))*f)]);
        end
        % Berechne den finalen Funktionswert f.
        f = eval(sprintf('D%d',num))*f;
        app = app + W(1,l)*f^2 - 2*W(2,l)*f;
    end
    app_total(1,t) = app;

    % Stückweise Berechnung von err.
    err_1 = 0;
    err_2 = 0; 
    for n = 1:N
        % Initialisiere den Funktionswert f für das NN.
        f = X(:,n);
        for lay = 1:num-1
            % Werte die Funktion f des NN stückweise aus.
            f = sigmoid([dlarray(eval(sprintf('D%d',lay))*f)]);
        end
        % Berechne den finalen Funktionswert f.
        f = eval(sprintf('D%d',num))*f;
        err_1 = err_1 + f^2;
        err_2 = err_2 + Y(n)*f;
    end
    err = 1/N * err_1 - 2/N * err_2;

    % Speichere den Fehler ab.
    err_total(1,t) = err;
    err_app(1,t) = abs(err-app);
end

end