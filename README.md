# HMI enhancer
Enhance is a deep learning technique for deconvolving and superresolving HMI continuum images and magnetograms.

train.py
--------

Example

python train.py --output=networks/test --epochs=20 --depth=5 --kernels=64 --action=start --model=keepsize --activation=relu --lr=1e-4 --lr_multiplier=1.0 --batchsize=32 --l2_regularization=1e-8

    --action={start,continue}
        `start`: start a new calculation
        `continue`: continue a previous calculation
    --epochs=20
        Number of epochs to use during training
    --output=networks/keepsize_relu 
        Define the output file that will contain the network topology and weights
    --depth=5
        Number of residual blocks used in the network. This number will affect differently depending on the topology of the network
    --model={keepsize,encdec}
        `keepsize` is a network that maintains the size of the input and output, with an eventual upsampling at the end in case of superresolution
        `encdec` is an encoder-decoder network
    --padding={zero,reflect}
        `zero` uses zero padding for keeping the size of the images through the network. This might produce some border artifacts 
        `reflect` uses reflection padding, which strongly reduces these artifacts
    --activation={relu,elu}
        Type of activation function to be used in the network, except for the last convolutional layer, which uses a linear activation