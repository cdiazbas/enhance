import numpy as np
from astropy.io import fits
import platform
import os
import json
import argparse
import h5py
from contextlib import redirect_stdout
import copy
from ipdb import set_trace as stop

if (platform.node() == 'viga'):
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'viga'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import time
import models as nn_model

def flush_file(f):
    f.flush()
    os.fsync(f.fileno())    

class LossHistory(Callback):
    def __init__(self, root, depth, losses, extra, **kwargs):        
        self.losses = losses
        self.losses_batch = copy.deepcopy(losses)
        self.extra = extra

        self.f_epoch = open("/net/vena/scratch/Dropbox/GIT/DeepLearning/losses/{0}_loss.json".format(platform.node()), 'w')
        self.f_batch = open("/net/vena/scratch/Dropbox/GIT/DeepLearning/losses/{0}_loss_batch.json".format(platform.node()), 'w')
        self.f_epoch.write('['+json.dumps(self.extra))
        self.f_batch.write('['+json.dumps(self.extra))

        self.f_epoch_local = open("{0}_{1}_loss.json".format(root, depth), 'w')
        self.f_batch_local = open("{0}_{1}_loss_batch.json".format(root, depth), 'w')
        self.f_epoch_local.write('['+json.dumps(self.extra))
        self.f_batch_local.write('['+json.dumps(self.extra))

        flush_file(self.f_batch)
        flush_file(self.f_batch_local)
        flush_file(self.f_epoch)
        flush_file(self.f_epoch_local)

    def on_batch_end(self, batch, logs={}):
        tmp = [time.asctime(),logs.get('loss').tolist(), ktf.get_value(self.model.optimizer.lr).tolist()]
        self.f_batch.write(','+json.dumps(tmp))
        self.f_batch_local.write(','+json.dumps(tmp))

        flush_file(self.f_batch)
        flush_file(self.f_batch_local)

    def on_epoch_end(self, batch, logs={}):
        tmp = [time.asctime(),logs.get('loss').tolist(), logs.get('val_loss').tolist(), ktf.get_value(self.model.optimizer.lr).tolist()]
        self.f_epoch.write(','+json.dumps(tmp))
        self.f_epoch_local.write(','+json.dumps(tmp))

        flush_file(self.f_epoch)
        flush_file(self.f_epoch_local)
        
    def on_train_end(self, logs):
        self.f_batch.write(']')
        self.f_batch_local.write(']')
        self.f_epoch.write(']')
        self.f_epoch_local.write(']')

        self.f_batch.close()
        self.f_epoch.close()
        self.f_batch_local.close()
        self.f_epoch_local.close()

    def finalize(self):
        pass

class deep_network(object):

    def __init__(self, root, noise, option, depth, network_type, activation, lr, lr_multiplier, batch_size, nkernels):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)


        self.root = root
        self.option = option
        self.noise = noise
        self.depth = depth
        self.n_filters = nkernels
        self.network_type = network_type        
        self.activation = activation        
        self.lr = lr
        self.lr_multiplier = lr_multiplier
        self.batch_size = batch_size

        tmp = np.loadtxt('/net/vena/scratch/Dropbox/GIT/DeepLearning/hmi_super/training/normalization.txt')
        self.median_HMI, self.median_SST = tmp[0], tmp[1]        

        self.input_file_images_training = "/net/viga/scratch1/cdiazbas/DATABASE/database_training_x2_PSF2.h5"
        
        f = h5py.File(self.input_file_images_training, 'r')
        self.n_training_orig, self.nx, self.ny, _ = f['imHMI'].shape        
        f.close()

        self.input_file_images_validation = "/net/viga/scratch1/cdiazbas/DATABASE/database_validation_x2_PSF2.h5"
        
        f = h5py.File(self.input_file_images_validation, 'r')
        self.n_validation_orig, self.nx, self.ny, _ = f['imHMI'].shape        
        f.close()        
        
        self.batchs_per_epoch_training = int(self.n_training_orig / self.batch_size)
        self.batchs_per_epoch_validation = int(self.n_validation_orig / self.batch_size)

        self.n_training = self.batchs_per_epoch_training * self.batch_size
        self.n_validation = self.batchs_per_epoch_validation * self.batch_size

        print("Original training set size: {0}".format(self.n_training_orig))
        print("   - Final training set size: {0}".format(self.n_training))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_training))

        print("Original validation set size: {0}".format(self.n_validation_orig))
        print("   - Final validation set size: {0}".format(self.n_validation))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_validation))

    def training_generator(self):
        f = h5py.File(self.input_file_images_training, 'r')
        
        while 1:        
            for i in range(self.batchs_per_epoch_training):

                input_train = f['imHMI'][i*self.batch_size:(i+1)*self.batch_size,:,:,0:1].astype('float32') / self.median_HMI
                output_train = f['imSST'][i*self.batch_size:(i+1)*self.batch_size,:,:,0:1].astype('float32') / self.median_SST

                yield input_train, output_train

        f.close()

    def validation_generator(self):
        f = h5py.File(self.input_file_images_validation, 'r')
        
        while 1:        
            for i in range(self.batchs_per_epoch_validation):

                input_validation = f['imHMI'][i*self.batch_size:(i+1)*self.batch_size,:,:,0:1].astype('float32') / self.median_HMI
                output_validation = f['imSST'][i*self.batch_size:(i+1)*self.batch_size,:,:,0:1].astype('float32') / self.median_SST

                yield input_validation, output_validation

        f.close()
        

    def define_network(self, l2_reg):
        print("Setting up network...")

        if (self.network_type == 'encdec'):
            self.model = nn_model.encdec(self.nx, self.ny, self.noise, self.depth, activation=self.activation, n_filters=self.n_filters)

        if (self.network_type == 'keepsize'):
            self.model = nn_model.keepsize(self.nx, self.ny, self.noise, self.depth, activation=self.activation, n_filters=self.n_filters, l2_reg=l2_reg)
        
            
        json_string = self.model.to_json()
        f = open('{0}_{1}_model.json'.format(self.root, self.depth), 'w')
        f.write(json_string)
        f.close()

        with open('{0}_{1}_summary.txt'.format(self.root, self.depth), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        plot_model(self.model, to_file='{0}_{1}_model.png'.format(self.root, self.depth), show_shapes=True)

    
    def compile_network(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        
    def read_network(self):
        print("Reading previous network...")

        if (self.network_type == 'encdec'):
            self.model = nn_model.encdec(self.nx, self.ny, self.noise, self.depth, activation=self.activation, n_filters=self.n_filters)

        if (self.network_type == 'keepsize'):
            self.model = nn_model.keepsize(self.nx, self.ny, self.noise, self.depth, activation=self.activation, n_filters=self.n_filters)

        self.model.load_weights("{0}_{1}_weights.hdf5".format(self.root, self.depth))

    def learning_rate(self, epoch):
        value = self.lr
        if (epoch >= 20):
            value *= self.lr_multiplier
        return value

    def train(self, n_iterations):
        print("Training network...")        
        
        # Recover losses from previous run
        if (self.option == 'continue'):
            with open("{0}_{1}_loss.json".format(self.root, self.depth), 'r') as f:
                losses = json.load(f)
        else:
            losses = []

        self.checkpointer = ModelCheckpoint(filepath="{0}_{1}_weights.hdf5".format(self.root, self.depth), verbose=1, save_best_only=True)
        self.history = LossHistory(self.root, self.depth, losses, {'name': '{0}_{1}'.format(self.root, self.depth), 'init_t': time.asctime()})

        self.reduce_lr = LearningRateScheduler(self.learning_rate)
        
        self.metrics = self.model.fit_generator(self.training_generator(), self.batchs_per_epoch_training, epochs=n_iterations, 
            callbacks=[self.checkpointer, self.history, self.reduce_lr], validation_data=self.validation_generator(), validation_steps=self.batchs_per_epoch_validation)
        
        self.history.finalize()

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train/predict for MFBD')
    parser.add_argument('-o','--output', help='Output files')
    parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
    parser.add_argument('-n','--noise', help='Noise to add during training/prediction', default=0.0)
    parser.add_argument('-d','--depth', help='Depth', default=5)
    parser.add_argument('-k','--kernels', help='N. kernels', default=64)
    parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], required=True)
    parser.add_argument('-m','--model', help='Model', choices=['encdec', 'keepsize'], required=True, default='keepsize')    
    parser.add_argument('-c','--activation', help='Activation', choices=['relu', 'elu'], required=True, default='relu')
    parser.add_argument('-lr','--lr', help='Learning rate', required=True, default=1e-4)
    parser.add_argument('-lrm','--lr_multiplier', help='Learning rate multiplier', required=True, default=0.96)
    parser.add_argument('-l2','--l2_regularization', help='L2 regularization', required=False, default=1e-7)
    parser.add_argument('-b','--batchsize', help='Batch size', required=True, default=32)
    parsed = vars(parser.parse_args())

    root = parsed['output']
    nEpochs = int(parsed['epochs'])
    option = parsed['action']
    network_type = parsed['model']    
    noise = float(parsed['noise'])
    depth = int(parsed['depth'])
    activation = parsed['activation']
    lr = float(parsed['lr'])
    lr_multiplier = float(parsed['lr_multiplier'])
    batch_size = int(parsed['batchsize'])
    nkernels = int(parsed['kernels'])

# Save parameters used
    with open("{0}_{1}_args.json".format(root, depth), 'w') as f:
        json.dump(parsed, f)

    out = deep_network(root, noise, option, depth, network_type, activation, lr, lr_multiplier, batch_size, nkernels)

    if (option == 'start'):           
        out.define_network(float(parsed['l2_regularization']))        
        
    if (option == 'continue' or option == 'predict'):
        out.read_network()

    if (option == 'start' or option == 'continue'):
        out.compile_network()
        out.train(nEpochs)
