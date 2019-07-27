import numpy as np
import platform
import os
import time
import argparse
from astropy.io import fits

# To deactivate warnings: https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

import models as nn_model

class enhance(object):

    def __init__(self, inputFile, depth, model, activation, ntype, output):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)
        self.hdu = fits.open(inputFile)
        #Autofix broken header files according to fits standard
        self.hdu.verify('silentfix')
        self.image = self.hdu[0].data
        self.header = self.hdu[0].header

        self.input = inputFile
        self.depth = depth
        self.network_type = model
        self.activation = activation
        self.ntype = ntype
        self.output = output


    def define_network(self): #, image):
        print("Setting up network...")

        #self.image = image
        image = self.image
        self.nx = image.shape[1]
        self.ny = image.shape[0]

        if (self.network_type == 'encdec'):
            self.model = nn_model.encdec(self.ny, self.nx, 0.0, self.depth, n_filters=64)

        # if (self.network_type == 'encdec_reflect'):
        #     self.model = nn_model.encdec_reflect(self.nx, self.ny, 0.0, self.depth, n_filters=64)

        # if (self.network_type == 'keepsize_zero'):
        #     self.model = nn_model.keepsize_zero(self.nx, self.ny, 0.0, self.depth)

        if (self.network_type == 'keepsize'):
            self.model = nn_model.keepsize(self.ny, self.nx, 0.0, self.depth,n_filters=64, l2_reg=1e-7)

        print("Loading weights...")
        self.model.load_weights("network/{0}_weights.hdf5".format(self.ntype))


    def predict(self):
        print("Predicting validation data...")

        input_validation = np.zeros((1,self.ny,self.nx,1), dtype='float32')
        input_validation[0,:,:,0] = self.image

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))

        print("Updating header")
        #Calculate scale factor (currently should be 0.5 because of 2 factor upscale)
        new_data = out[0,:,:,0]
        new_dim = new_data.shape

        scale_factor_x = float(self.nx / new_dim[1])
        scale_factor_y = float(self.ny / new_dim[0])

        #fix map scale after upsampling
        if 'cdelt1' in self.header:
            self.header['cdelt1'] *= scale_factor_x
            self.header['cdelt2'] *= scale_factor_y

        #WCS rotation keywords used by IRAF and HST
        if 'CD1_1' in self.header:
            self.header['CD1_1'] *= scale_factor_x
            self.header['CD2_1'] *= scale_factor_x
            self.header['CD1_2'] *= scale_factor_y
            self.header['CD2_2'] *= scale_factor_y

        #Patch center with respect of lower left corner
        if 'crpix1' in self.header:
            self.header['crpix1'] = (new_dim[1] + 1) / 2.
            self.header['crpix2'] = (new_dim[0] + 1) / 2.

        #Number of pixel per axis
        if 'naxis1' in self.header:
            self.header['naxis1'] /= scale_factor_x
            self.header['naxis2'] /= scale_factor_y

        print("Saving data...")
        hdu = fits.PrimaryHDU(new_data, self.header)
        import os.path
        if os.path.exists(self.output):
            os.system('rm {0}'.format(self.output))
            print('Overwriting...')
        hdu.writeto('{0}'.format(self.output))

        # import matplotlib.pyplot as plt
        # plt.imshow(out[0,:,:,0])
        # plt.savefig('hmi.pdf')


if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('-i','--input', help='input')
    parser.add_argument('-o','--out', help='out')
    parser.add_argument('-d','--depth', help='depth', default=5)
    parser.add_argument('-m','--model', help='model', choices=['encdec', 'encdec_reflect', 'keepsize_zero', 'keepsize'], default='keepsize')
    parser.add_argument('-c','--activation', help='Activation', choices=['relu', 'elu'], default='relu')
    # parser.add_argument('-a','--action', help='action', choices=['cube', 'movie'], default='cube')
    parser.add_argument('-t','--type', help='type', choices=['intensity', 'blos'], default='intensity')
    parsed = vars(parser.parse_args())

    #f = fits.open(parsed['input'])
    #imgs = f[0].data
    #hdr = f[0].header

    print('Model : {0}'.format(parsed['type']))
    out = enhance('{0}'.format(parsed['input']), depth=int(parsed['depth']), model=parsed['model'], activation=parsed['activation'],ntype=parsed['type'], output=parsed['out'])
    #out.define_network(image=imgs)
    out.define_network()
    out.predict()
    # To avoid the TF_DeleteStatus message:
    # https://github.com/tensorflow/tensorflow/issues/3388
    ktf.clear_session()

    # python enhance.py -i samples/hmi.fits -t intensity -o output/hmi_enhanced.fits

    # python enhance.py -i samples/blos.fits -t blos -o output/blos_enhanced.fits
