import numpy as np
import platform
import os
import time
import argparse
from astropy.io import fits
import tensorflow as tf
import keras as krs
import models as nn_model

# Using TensorFlow backend
os.environ["KERAS_BACKEND"] = "tensorflow"

print('tensorflow version:',tf.__version__)
print('keras version:',krs.__version__)

class enhance(object):

    def __init__(self, inputFile, depth, model, activation, ntype, output):

        self.hdu = fits.open(inputFile)
        #Autofix broken header files according to fits standard
        self.hdu.verify('silentfix')
        index = 0
        if np.all(self.hdu[index].data == None): index = 1
        self.image = np.nan_to_num(self.hdu[index].data[:,:])
        self.header = self.hdu[index].header
        print('Size image: ',self.image.shape)

        self.input = inputFile
        self.depth = depth
        self.network_type = model
        self.activation = activation
        self.ntype = ntype
        self.output = output
        self.big_image = 2048
        self.split = False
        self.norm = 1.0

        if self.ntype == 'intensity': 
            self.norm = np.max(self.image)
        if self.ntype == 'blos': 
            self.norm = 1e3
        
        self.image = self.image/self.norm

    def define_network(self): #, image):
        print("Setting up network...")
        #self.image = image
        self.nx = self.image.shape[1]
        self.ny = self.image.shape[0]
        if self.nx > self.big_image or self.ny > self.big_image:
            self.split = True
            self.nx = int(self.image.shape[1]/2)
            self.ny = int(self.image.shape[0]/2)

        if (self.network_type == 'keepsize'):
            self.model = nn_model.keepsize(self.ny, self.nx, 0.0, self.depth,n_filters=64, l2_reg=1e-7)

        print("Loading weights...")
        self.model.load_weights("network/{0}_weights.hdf5".format(self.ntype))

    def predict_image(self,inputdata):
        # Patch for big images in keras
        if self.split is True: 
            M = inputdata.shape[1]//2
            N = inputdata.shape[2]//2
            out = np.empty((1,inputdata.shape[1]*2,inputdata.shape[2]*2, 1))
            for x in range(0,inputdata.shape[1],M):
                for y in range(0,inputdata.shape[2],N):
                    out[:,x*2:x*2+M*2,y*2:y*2+N*2,:] = self.model.predict(inputdata[:,x:x+M,y:y+N,:])
            
            self.nx = inputdata.shape[2]
            self.ny = inputdata.shape[1]

        else:
            out = self.model.predict(inputdata)
        
        print('New size: ',out.shape)
        return out
    
    def predict(self,plot_option=False,sunpy_map=False):
        print("Predicting data...")

        input_validation = np.zeros((1,self.image.shape[0],self.image.shape[1],1), dtype='float32')
        input_validation[0,:,:,0] = self.image

        start = time.time()
        out = self.predict_image(input_validation)
        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))

        print("Updating header ...")
        # Calculate scale factor (currently should be 0.5 because of 2 factor upscale)
        new_data = out[0,:,:,0]
        new_data = new_data*self.norm
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
            self.header['crpix1'] /= scale_factor_x
            self.header['crpix2'] /= scale_factor_y

        #Number of pixel per axis
        if 'naxis1' in self.header:
            self.header['naxis1'] = new_dim[1]
            self.header['naxis2'] = new_dim[0]


        print("Saving data...")
        hdu = fits.PrimaryHDU(new_data, self.header)
        import os.path
        if os.path.exists(self.output):
            os.system('rm {0}'.format(self.output))
            print('Overwriting...')
        hdu.writeto('{0}'.format(self.output), output_verify="ignore")

        if plot_option is True:
            print("Plotting...")
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.image,cmap='gray',origin='lower',vmin=self.image.min(),vmax=self.image.max())
            plt.subplot(122)
            plt.imshow(out[0,:,:,0],cmap='gray',origin='lower',vmin=self.image.min(),vmax=self.image.max())
            plt.tight_layout()
            plt.savefig('hmi_test.pdf', bbox_inches='tight')

            if sunpy_map is True:
                import sunpy.map
                sdomap0 =sunpy.map.Map(self.input,self.header)
                sdomap1 =sunpy.map.Map(self.output,self.header)
                plt.figure()
                plt.subplot(121)
                sdomap0.plot()
                plt.subplot(122)
                sdomap1.plot()
                plt.tight_layout()
                plt.savefig('hmi_test2.pdf', bbox_inches='tight')


if (__name__ == '__main__'):

    """
    Using Enhance for prediction:
    =============================

    python enhance.py -i samples/hmi.fits -t intensity -o output/hmi_enhanced.fits

    python enhance.py -i samples/blos.fits -t blos -o output/blos_enhanced.fits

    """

    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('-i','--input', help='input')
    parser.add_argument('-o','--out', help='out')
    parser.add_argument('-d','--depth', help='depth', default=5)
    parser.add_argument('-m','--model', help='model', choices=['encdec', 'encdec_reflect', 'keepsize_zero', 'keepsize'], default='keepsize')
    parser.add_argument('-c','--activation', help='Activation', choices=['relu', 'elu'], default='relu')
    parser.add_argument('-t','--type', help='type', choices=['intensity', 'blos'], default='intensity')
    parsed = vars(parser.parse_args())


    print('Model : {0}'.format(parsed['type']))
    out = enhance('{0}'.format(parsed['input']), depth=int(parsed['depth']), model=parsed['model'], activation=parsed['activation'],ntype=parsed['type'], output=parsed['out'])
    out.define_network()
    out.predict()
 

