function [loss,gradients] = modelLoss(net,X,Y,W1,W2,type)

if strcmp(type,'err') == 0 && strcmp(type,'app') == 0
    fprintf('Die Eingabe von type ist unzulässig.\n')
    return
end

% Forward data through network.
[X] = forward(net,X);

% Calculate loss and gradients.
if strcmp(type,'err') == 1
    % Calculate cross-entropy loss.
    loss = 2*mse(X,Y);
    % Calculate gradients of loss with respect to learnable parameters.
    [gradients] = dlgradient(loss,net.Learnables);
else
    % Calculate cross-entropy loss.
    len = size(X,2);
    loss = 0;
    for l = 1:len
        loss = loss + X(l)^2*W1(l) - 2*X(l)*W2(l) + Y;
    end
    % Calculate gradients of loss with respect to learnable parameters.
    [gradients] = dlgradient(loss,net.Learnables);
end

end
