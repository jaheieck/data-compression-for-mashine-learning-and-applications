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


%% compress data with the supercompress method
% definitions of the parmeters and generation of the random numbers
tic % measure time spent
K = 2^12;
s = size(XTrain,1)^2;

XTrain_reshape = reshape(XTrain,[],size(XTrain,4)); % reshape for the supercompress

% calculate supercom data compression for the normal supercompress
[x_supercom,y_supercom] = supercompress_NN_image(XTrain_reshape,YTrain,K,1); 
x_supercom_Train = reshape(x_supercom,sqrt(s),sqrt(s),1,K); % transofrm data into correct dimension
y_supercom_Train = categorical(round(y_supercom)'); % round the cluster response and transform them to categorical


%% train the neural network with original and compressed data - define necessary input
pixelRange = [-3 3];
imageSize = [size(XTrain,1) size(XTrain,1) 1];
imageAugmenter = imageDataAugmenter('RandRotation',[-20,20],'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,x_supercom_Train,y_supercom_Train,'DataAugmentation',imageAugmenter);
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

numObservationsTrain = size(x_supercom_Train,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;


%% Loop over epochs for standard data
monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");

epoch = 0;
iteration = 0;
while epoch < numEpochs && ~monitor.Stop
    
    epoch = epoch + 1;

    % shuffle data
    shuffle(mbq);
    
    % loop over mini-batches
    while hasdata(mbq) && ~monitor.Stop

        iteration = iteration + 1;
        
        % read mini-batch of data
        [X,T] = next(mbq);
        
        % evaluate the model gradients, state, and loss using dlfeval and the
        % modelLoss function and update the network state
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
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







function [loss,gradients,state] = modelLoss(net,X,T)
% forward data through network
[Y,state] = forward(net,X);
% calculate cross-entropy loss
loss = mse(Y,T);
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
% one-hot encode labels
T = onehotencode(T,1);
end

function X = preprocessMiniBatchPredictors(dataX)
% concatenate
X = cat(4,dataX{1:end});
end