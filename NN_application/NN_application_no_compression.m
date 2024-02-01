%% load the necessary files, functions and data
clear
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

tic % measure time spent


%% train the neural network with original and compressed data - define necessary input
pixelRange = [-3 3];
imageSize = [size(XTrain,1) size(XTrain,1) 1];
imageAugmenter = imageDataAugmenter('RandRotation',[-20,20],'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);
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

numObservationsTrain = size(XTrain,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;


%% Loop over epochs for standard data
monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");

epoch = 0;
iteration = 0;
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
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
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

time = toc;

numOutputs = 1;
mbqTest = minibatchqueue(augimdsTest,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");

YPred = modelPredictions(net,mbqTest,classes);
accuracy = mean(YTest' == YPred);

figure
confusionchart(YTest,YPred)







function [loss,gradients,state] = modelLoss(net,X,T)
% Forward data through network.
[Y,state] = forward(net,X);
% Calculate cross-entropy loss.
loss = mse(Y,T);
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


function [X,T] = preprocessMiniBatch(dataX,dataT)
% Preprocess predictors.
X = preprocessMiniBatchPredictors(dataX);
% Extract label data from cell and concatenate.
T = cat(2,dataT{1:end});
% One-hot encode labels.
T = onehotencode(T,1);
end

function X = preprocessMiniBatchPredictors(dataX)
% Concatenate.
X = cat(4,dataX{1:end});
end