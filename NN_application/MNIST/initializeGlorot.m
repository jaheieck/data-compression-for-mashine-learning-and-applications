% Die Funktion wurde aus https://de.mathworks.com/help/deeplearning/ug/train-a-variational-autoencoder-vae-to-generate-images.html
% übernommen.
function weights = initializeGlorot(sz,numOut,numIn,className)

arguments
    sz
    numOut
    numIn
    className = 'single'
end

Z = 2*rand(sz,className) - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end