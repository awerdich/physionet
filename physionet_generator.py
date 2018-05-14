#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physionet ECG classification

@author: Andreas Werdich

Batch generator class
Modified from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

"""
#%% Imports

import numpy as np
import keras

from physionet_processing import (zero_filter, extend_ts, 
                                  random_resample, spectrogram, norm_float)

#%% Batch generator class

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, h5file, list_IDs, labels, batch_size = 32, dim = (178, 33), 
                 nperseg = 64, noverlap = 32, data_mean = -9.01, data_std = 9.00,  
                 n_channels=1, sequence_length = 5736, 
                 n_classes = 4, shuffle = True, augment = False):
        
        'Initialization'
        self.h5file = h5file
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.data_mean = data_mean
        self.data_std = data_std
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = float)
        y = np.empty((self.batch_size), dtype = int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            data = extend_ts(self.h5file[ID]['ecgdata'][:, 0], self.sequence_length)
            data = np.reshape(data, (1, len(data)))
        
            if self.augment:
            
                # dropout bursts
                data = zero_filter(data, threshold = 2, depth = 10)
            
                # random resampling
                data = random_resample(data)
            
            # Generate spectrogram
            data_spectrogram = spectrogram(data, nperseg = self.nperseg, noverlap = self.noverlap)[2]
            
            # Normalize
            data_transformed = norm_float(data_spectrogram, self.data_mean, self.data_std)
        
            X[i,] = np.expand_dims(data_transformed, axis = 3)
        
            # Assuming that the dataset names are unique (only 1 per label)
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
 

# Run as Script

if __name__ == '__main__':
    pass