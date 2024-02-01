% This function is taken from https://de.mathworks.com/help/deeplearning/ug/train-a-variational-autoencoder-vae-to-generate-images.html
function parameter = initializeZeros(sz,className)

arguments
    sz
    className = 'single'
end

parameter = zeros(sz,className);
parameter = dlarray(parameter);

end