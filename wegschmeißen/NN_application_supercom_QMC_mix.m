%% load the necessary files, functions and data
clear
addpath(genpath([pwd, filesep, '\MNIST']));
addpath(genpath([fileparts(pwd), filesep, '\QMCpoints']));
addpath(genpath([fileparts(pwd), filesep, '\QMCweights']));
XTrain = processImagesMNIST('train-images-idx3-ubyte.gz');
YTrain = processLabelsMNIST('train-labels-idx1-ubyte.gz');
XTest = processImagesMNIST('t10k-images-idx3-ubyte.gz');
YTest = processLabelsMNIST('t10k-labels-idx1-ubyte.gz');

% precompress data to reduce running time
XTrain = XTrain(:,:,:,1:10^4);
YTrain = YTrain(1,1:10^4);
[XTrain] = NN_stencil(XTrain,2);
[XTest] = NN_stencil(XTest,2);


%% cluster XTrain based on their x-values with kmeans
tic % measure time spent
XTrain = NN_stencil(XTrain,4);
XTrain_reshape = reshape(XTrain,[],size(XTrain,4)); % reshape the data to fit the kmeans
K = 200; % number of clusters
[idx_temp,x_temp] = kmeans(XTrain_reshape',K);

%% Calculate the weights for each cluster seperately 

% QMC Voronoi
b = 2;
for k = 1:K
    X = XTrain_reshape(:,idx_temp == k);
    X = X(sum(X,2) ~= 0,:);
    X = X./max(X,[],2);
    size(X)
    Y = YTrain(idx_temp == k);
    Y = cellstr(Y);
    Y = str2double(Y);
    N = size(X,2);
    s = size(X,1);
    m = 12;
    [QMC.(strcat('P_',num2str(k))),QMC.(strcat('Yp_',num2str(k)))] = QMC_Voronoicompress(b,m,X,Y);
    round(k/K*100,1)
end

% normal QMC
% b = 2;
% for k = 1:K
%     X = XTrain_reshape(:,idx_temp == k);
%     Y = YTrain(idx_temp == k);
%     Y = cellstr(Y);
%     Y = str2double(Y);
%     N = size(X,2);
%     s = size(X,1);
%     m = 4;
%     L = b^m;
%     nu = 2;
%     load 'DIGSEQ\sobolmats\Sobol_Cs.col'
%     digitalseq_b2g('init0', Sobol_Cs)
%     QMC.(strcat('P_',num2str(k))) = digitalseq_b2g(s,L);
%     W = zeros(2,L);
%     for l = 1:L % calculate weights
%        W(1,l) = weight(QMC.(strcat('P_',num2str(k)))(:,l),X,ones(1,N),b,m,nu);
%        W(2,l) = weight(QMC.(strcat('P_',num2str(k)))(:,l),X,Y,b,m,nu);
%     end
%     QMC.(strcat('W_',num2str(k))) = W;
%     k
% end


%% combine the point sets and transform data to desired structure
% PTrain = [];
% Yp = [];
% for k = 1:K
%     PTrain = [PTrain QMC.(strcat('P_',num2str(k)))];
%     Yp = [Yp QMC.(strcat('Yp_',num2str(k)))];
% end
% P_QMCVor_Train = reshape(PTrain,size(XTrain,1),size(XTrain,1),1,[]);
% Y_QMCVor = Yp(1,:)';
% Y_QMCVor_Train = categorical(round(Y_QMCVor)'); % round the cluster response and transform them to categorical

PTrain = QMC_Voronoicompress(b,m,XTrain_reshape,double(YTrain)-1);
Yp = zeros(2,b^m);
for k = 1:K
    Yp = Yp + QMC.(strcat('Yp_',num2str(k)));
end
P_QMCVor_Train = PTrain(:,Yp(2,:)~=0);
P_QMCVor_Train = reshape(P_QMCVor_Train,size(XTrain,1),size(XTrain,1),1,[]);
Yp(1,Yp(2,:)~=0) = Yp(1,Yp(2,:)~=0)./Yp(2,Yp(2,:)~=0);
Yp = Yp(:,Yp(2,:)~=0);
Y_QMCVor = Yp(1,:)';
Y_QMCVor_Train = categorical(round(Y_QMCVor)'); % round the cluster response and transform them to categorical

% %% compress data with the supercompress method
% % definitions of the parmeters and generation of the random numbers
% K = 0.1*size(XTrain,4);
% s = size(XTrain,1)^2;
% 
% XTrain_reshape = reshape(XTrain,[],size(XTrain,4)); % reshape the data to fit the kmeans
% % YTrain_help = double(YTrain)-1;
% 
% % calculate supercom data compression
% [x_kmeans,y_kmeans,~,idx_total] = supercompress_NN_image(XTrain_reshape,YTrain,K); 
% x_kmeans_Train = reshape(x_kmeans,sqrt(s),sqrt(s),1,K); % transofrm data into correct dimension
% y_kmeans_Train = categorical(round(y_kmeans)'); % round the cluster response and transform them to categorical
% 
% %% calculate QMCVoronoi compression
% [P_QMCVor,Yp] = QMC_Voronoicompress(b,m,XTrain_reshape,double(YTrain)-1);
% Y_QMCVor = Yp(:,1);
% P_QMCVor_Train = reshape(P_QMCVor,sqrt(s),sqrt(s),1,[]); % transofrm data into correct dimension
% Y_QMCVor_Train = categorical(round(Y_QMCVor)'); % round the cluster response and transform them to categorical



%% train the neural network with original and compressed data - define necessary input
pixelRange = [-3 3];
imageSize = [size(PTrain,1) size(PTrain,1) 1];
imageAugmenter = imageDataAugmenter('RandRotation',[-20,20],'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);
augimdsTrain_original = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);
% augimdsTrain_mixed= augmentedImageDatastore(imageSize,PTrain,W','DataAugmentation',imageAugmenter);
augimdsTrain_QMCVor = augmentedImageDatastore(imageSize,P_QMCVor_Train,Y_QMCVor_Train,'DataAugmentation',imageAugmenter);
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

numEpochs = 10;
miniBatchSize = 128;

initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

mbq_original = minibatchqueue(augimdsTrain_original,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB" ""]);

% mbq_mixed = minibatchqueue(augimdsTrain_mixed,...
%     MiniBatchSize=miniBatchSize,...
%     MiniBatchFcn=@preprocessMiniBatch,...
%     MiniBatchFormat=["SSCB" ""]);

mbq_QMCVor = minibatchqueue(augimdsTrain_QMCVor,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB" ""]);

velocity = [];

numObservationsTrain_original = size(XTrain,4);
% numObservationsTrain_mixed = size(PTrain,4);
numObservationsTrain_QMCVor = size(P_QMCVor_Train,4);
numIterationsPerEpoch_original = ceil(numObservationsTrain_original / miniBatchSize);
% numIterationsPerEpoch_mixed = ceil(numObservationsTrain_mixed / miniBatchSize);
numIterationsPerEpoch_QMCVor = ceil(numObservationsTrain_QMCVor / miniBatchSize);
numIterations_original = numEpochs * numIterationsPerEpoch_original;
% numIterations_mixed = numEpochs * numIterationsPerEpoch_mixed;
numIterations_QMCVor = numEpochs * numIterationsPerEpoch_QMCVor;


%% Loop over epochs for original data
monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");

epoch = 0;
iteration = 0;
while epoch < numEpochs && ~monitor.Stop
    
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq_original);
    
    % Loop over mini-batches.
    while hasdata(mbq_original) && ~monitor.Stop

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [X,T] = next(mbq_original);
        
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
        monitor.Progress = 100 * iteration/numIterations_original;
    end
end

numOutputs = 1;
mbqTest = minibatchqueue(augimdsTest,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");

YPred_original = modelPredictions(net,mbqTest,classes);
accuracy_original = mean(YTest' == YPred_original);

figure
confusionchart(YTest,YPred_original)


%% Loop over epochs for mixed compressed data
% monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");
% 
% epoch = 0;
% iteration = 0;
% while epoch < numEpochs && ~monitor.Stop
% 
%     epoch = epoch + 1;
% 
%     % Shuffle data.
%     shuffle(mbq_mixed);
% 
%     % Loop over mini-batches.
%     while hasdata(mbq_mixed) && ~monitor.Stop
% 
%         iteration = iteration + 1;
% 
%         % Read mini-batch of data.
%         [X,dataW] = next(mbq_mixed);
% 
%         % Evaluate the model gradients, state, and loss using dlfeval and the
%         % modelLoss function and update the network state.
%         [loss,gradients,state] = dlfeval(@modelLoss,net,X,dataW);
%         net.State = state;
% 
%         % Determine learning rate for time-based decay learning rate schedule.
%         learnRate = initialLearnRate/(1 + decay*iteration);
% 
%         % Update the network parameters using the SGDM optimizer.
%         [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
% 
%         % Update the training progress monitor.
%         recordMetrics(monitor,iteration,Loss=loss);
%         updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
%         monitor.Progress = 100 * iteration/numIterations_mixed;
%     end
% end
% 
% numOutputs = 1;
% mbqTest = minibatchqueue(augimdsTest,numOutputs, ...
%     MiniBatchSize=miniBatchSize, ...
%     MiniBatchFcn=@preprocessMiniBatchPredictors, ...
%     MiniBatchFormat="SSCB");
% 
% YPred_mixed = modelPredictions(net,mbqTest,classes);
% accuracy_mixed = mean(YTest' == YPred_mixed);
% 
% figure
% confusionchart(YTest,YPred_mixed)


%% Loop over epochs for mixed compressed data
monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");

epoch = 0;
iteration = 0;
while epoch < numEpochs && ~monitor.Stop
    
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq_QMCVor);
    
    % Loop over mini-batches.
    while hasdata(mbq_QMCVor) && ~monitor.Stop

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [X,T] = next(mbq_QMCVor);
        
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
        monitor.Progress = 100 * iteration/numIterations_QMCVor;
    end
end

time = toc;

numOutputs = 1;
mbqTest = minibatchqueue(augimdsTest,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");

YPred_QMCVor = modelPredictions(net,mbqTest,classes);
accuracy_QMCVor = mean(YTest' == YPred_QMCVor);

figure
confusionchart(YTest,YPred_QMCVor)







% function [loss,gradients,state] = modelLoss(net,X,dataW)
% % Forward data through network.
% [Y,state] = forward(net,X);
% % calculate loss
% len = size(Y,2);
% loss = 0;
% for l = 1:len
%     loss = loss + sum(Y(:,l).^2*dataW(1,l,1)) - sum(2*Y(:,l)*dataW(1,l,2)) + 10;
% end
% % Calculate gradients of loss with respect to learnable parameters.
% gradients = dlgradient(loss,net.Learnables);
% end


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