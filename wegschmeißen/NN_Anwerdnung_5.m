% Lade die notwendigen Ordner, die Funktionen beinhalten, die wir brauchen.
addpath(genpath([fileparts(pwd), filesep, '\Standardverfahren Berechnung der Gewichte']));
addpath(genpath([fileparts(pwd), filesep, '\Schrittweise Berechnung der Gewichte']));
addpath(genpath([fileparts(pwd), filesep, '\Fehlerberechnung Neuronales Netz']));
addpath(genpath([fileparts(pwd), filesep, '\Punkteberechnung']));
%addpath(genpath([fileparts(pwd), filesep, '\MNIST']));

XTrain = processImagesMNIST('train-images-idx3-ubyte.gz');
YTrain = processLabelsMNIST('train-labels-idx1-ubyte.gz');
XTest = processImagesMNIST('t10k-images-idx3-ubyte.gz');
YTest = processLabelsMNIST('t10k-labels-idx1-ubyte.gz');

% Komprimiere die Daten.
XTrain_komp = XTrain(:,:,:,1:6*10^2);
YTrain_komp = YTrain(1,1:6*10^2);
XTest_komp = XTrain(:,:,:,1:10^2);
YTest_komp = YTrain(1,1:10^2);
[XTrain_komp] = NN_stencil(XTrain_komp,2);
[XTest_komp] = NN_stencil(XTest_komp,2);
% Überführe YTrain_komp und Xtrain_komp in das richtige Format für die Gewichte.
YTrain_komp_help = cellstr(YTrain_komp);
YTrain_komp_help = str2double(YTrain_komp_help);
XTrain_komp_help = reshape(XTrain_komp,[],size(XTrain_komp,4));
% XTest_komp_help = reshape(XTest_komp,[],size(XTest_komp,4));

% Generiere das digitale Netz und berechne die Gewichte.
s = size(XTrain_komp,1)^2;
b = 2;
m = 10;
nu = 2;
L = b^m;
load 'DIGSEQ\sobolmats\sobol_Cs.col'
digitalseq_b2g('init0', sobol_Cs)
P = digitalseq_b2g(s,L);
% [W] = algorithm_5(P,XTrain_komp_help,YTrain_komp_help,b,m,nu);
W = ones(2,L);
tbl = [P' W(1,:)' W(2,:)'];
% PTrain = reshape(P,size(XTrain_komp,1),size(XTrain_komp,1),1,L);

classes = categories(YTest_komp);
Classes = [0 1 2 3 4 5 6 7 8 9];
% term = 1/size(YTrain_komp_help,2)* sum(yTrain_komp_help.^2);
% YTrain_komp(1,1:2^10)


pixelRange = [-3 3];
imageSize = [14 14 1];
imageAugmenter = imageDataAugmenter('RandRotation',[-20,20],'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,tbl,'responseNames','DataAugmentation',imageAugmenter);
augimdsTest = augmentedImageDatastore(imageSize,XTest_komp,YTest_komp,'DataAugmentation',imageAugmenter);


numclasses = length(classes);

layers = [
    imageInputLayer(imageSize,Normalization="none")
    convolution2dLayer(5,20,Padding="same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,20,Padding="same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,20,Padding="same")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numclasses)
    softmaxLayer];

net = dlnetwork(layers);

numEpochs = 10;
miniBatchSize = 128;

initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

mbq = minibatchqueue(augimdsTrain,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB" "" ""]);

velocity = [];

numObservationsTrain = size(XTrain_komp,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");

epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [X,T] = next(mbq);
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelLoss function and update the network state.
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,T,S);
        net.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
        
        % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
        monitor.Progress = 100 * iteration/numIterations;
    end
end

numOutputs = 1;
mbqTest = minibatchqueue(augimdsTest,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");

YPred = modelPredictions(net,mbqTest,classes);
accuracy = mean(YTest_komp == YPred);

figure
confusionchart(YTest_komp,YPred)

function [loss,gradients,state] = modelLoss(net,X,T,S)
% Forward data through network.
[Y,state] = forward(net,X);
% Calculate cross-entropy loss.
loss = 2*mse(Y,T) + 2*mse(Y,S);
% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss,net.Learnables);
end


function Y = modelPredictions(net,mbq,classes)
Y = [];
% Loop over mini-batches.
while hasdata(mbq)
    X = next(mbq);
    % Make prediction.
    scores = predict(net,X);
    % Decode labels and append to output.
    labels = onehotdecode(scores,classes,1)';
    Y = [Y; labels];
end
end


function [X,T,S] = preprocessMiniBatch(dataX,dataT,dataS)
% Preprocess predictors.
X = preprocessMiniBatchPredictors(dataX);
% Extract label data from cell and concatenate.
T = cat(2,dataT{1:end});
S = cat(2,dataS{1:end});
% One-hot encode labels.
T = onehotencode(categorical(T),1);
S = onehotencode(categorical(S),1);
end

function X = preprocessMiniBatchPredictors(dataX)
% Concatenate.
X = cat(4,dataX{1:end});
end