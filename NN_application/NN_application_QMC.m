%% load the necessary files, functions and data
clear 
addpath(genpath([fileparts(pwd), filesep, '\QMCpoints']));
addpath(genpath([fileparts(pwd), filesep, '\QMCweights']));
addpath(genpath([pwd, filesep, '\MNIST']));
XTrain = processImagesMNIST('train-images-idx3-ubyte.gz');
YTrain = processLabelsMNIST('train-labels-idx1-ubyte.gz');
XTest = processImagesMNIST('t10k-images-idx3-ubyte.gz');
YTest = processLabelsMNIST('t10k-labels-idx1-ubyte.gz');

% precompress data to reduce running time
XTrain = XTrain(:,:,:,1:10^4);
YTrain = YTrain(1,1:10^4);
[XTrain] = NN_stencil(XTrain,2);
[XTest] = NN_stencil(XTest,2);


%% compress data with QMC
tic % measure time spent
% transform YTrain and XTrain into the right data format for the weights
YTrain_help = cellstr(YTrain);
YTrain_help = str2double(YTrain_help);
XTrain_help = reshape(XTrain,[],size(XTrain,4));


%% construct digital net
s = size(XTrain,1)^2;
b = 2;
m = 10;
nu = 2;
L = b^m;
load 'DIGSEQ\sobolmats\sobol_Cs.col'
digitalseq_b2g('init0', sobol_Cs)
P = digitalseq_b2g(s,L);
[W] = algorithm_5(P,XTrain_help,YTrain_help,b,m,nu)';
PTrain = reshape(P,size(XTrain,1),size(XTrain,1),1,[]);

%% prepare neural network
pixelRange = [-3 3];
imageSize = [size(PTrain,1) size(PTrain,1) 1];
imageAugmenter = imageDataAugmenter('RandRotation',[-20,20],'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,PTrain,W,'DataAugmentation',imageAugmenter);
augimdsTest = augmentedImageDatastore(imageSize,XTest,YTest,'DataAugmentation',imageAugmenter);

classes = categories(YTest);
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

numEpochs = 100;
miniBatchSize = 128;

initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

mbq = minibatchqueue(augimdsTrain,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB" ""]);

velocity = [];

numObservationsTrain = size(PTrain,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");

epoch = 0;
iteration = 0;

%% loop over epochs
while epoch < numEpochs && ~monitor.Stop
    
    epoch = epoch + 1;

    % shuffle data
    shuffle(mbq);
    
    % loop over mini-batches
    while hasdata(mbq) && ~monitor.Stop

        iteration = iteration + 1;
        
        % read mini-batch of data
        [X,dataW] = next(mbq);
        
        % evaluate the model gradients, state, and loss using dlfeval and the
        % modelLoss function and update the network state
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,dataW);
        net.State = state;
        
        % determine learning rate for time-based decay learning rate schedule
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % update the network parameters using the SGDM optimizer
        [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
        
        % update the training progress monitor
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
        monitor.Progress = 100 * iteration/numIterations;
    end
end

time = toc;

% test the neural network on the test data
numOutputs = 1;
mbqTest = minibatchqueue(augimdsTest,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");

YPred = modelPredictions(net,mbqTest,classes);
% evaluate accuracy of the model 
accuracy = mean(YTest' == YPred);

% print the confusion chart
figure
confusionchart(YTest,YPred)









function [loss,gradients,state] = modelLoss(net,X,dataW)
% forward data through network
[Y,state] = forward(net,X);
% calculate loss
len = size(Y,2);
loss = 0;
for l = 1:len
    loss = loss + sum(Y(:,l).^2*dataW(1,l,1)) - sum(2*Y(:,l)*dataW(1,l,2)) + 10;
end
% calculate gradients of loss with respect to learnable parameters
gradients = dlgradient(loss,net.Learnables);
end


function Y = modelPredictions(net,mbq,classes)
Y = [];
% loop over mini-batches
while hasdata(mbq)
    X = next(mbq);
    % make prediction
    scores = predict(net,X);
    % decode labels and append to output
    labels = onehotdecode(scores,classes,1)';
    Y = [Y; labels];
end
end


function [X,T] = preprocessMiniBatch(dataX,dataT)
% preprocess predictors
X = preprocessMiniBatchPredictors(dataX);
% extract label data from cell and concatenate
T = cat(2,dataT{1:end});
% one-hot encode labels - only for k-means
% T = onehotencode(T,1);
end

function X = preprocessMiniBatchPredictors(dataX)
% concatenate
X = cat(4,dataX{1:end});
end


