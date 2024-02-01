% Hier Visualisieren wir die Asuwirkung des Komprimierungsansatzes stencil, 
% um besser klarzustellen, was eine Verringerung der Dimension der 
% Eingabedaten für eine Auswirkung auf die Daten hat.

% Lade die notwendigen Ordner, die Funktionen beinhalten, die wir brauchen.
addpath(genpath([pwd, filesep, '\MNIST']));

% Daten laden von https://yann.lecun.com/exdb/mnist/.
XTrain = processImagesMNIST('train-images-idx3-ubyte.gz');

% Komprimiere die Daten.
[XTrain_komp1] = NN_stencil(XTrain,2);
[XTrain_komp2] = NN_stencil(XTrain,4);

% Plots zu den Graubildern.
Komprimierung = figure;
subplot(1,3,1)
I_1 = imtile(XTrain(:,:,1,1:16));
imshow(I_1)
title('28x28-Matrix')
subplot(1,3,2)
I_2 = imtile(XTrain_komp1(:,:,1,1:16));
imshow(I_2)
title('14x14-Matrix')
subplot(1,3,3)
I_3 = imtile(XTrain_komp2(:,:,1,1:16));
imshow(I_3)
title('7x7-Matrix')

% Speicherung der Daten.
% saveas(Komprimierung,'Komprimierung.png')
