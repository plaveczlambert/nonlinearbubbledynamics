'''Transform bubble dynamics leraning data generated for normal DEM integration into temperature data, 
useful for training neural network of the hybrid method. The script copies the data while also removing 
unnecessary columns of output data'''

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.integrate
import h5py

N_z = 7

path_to_hdf = '../data/data0.5.hdf5'

new_hdf = '../data/data0.5_hybrid.hdf5'


with h5py.File(path_to_hdf, 'r') as f:
    x = f["bub_X"]
    y = f["bub_Y"]
    print(np.max(x[:,0]))
    print(np.min(x[:,0]))
    print(np.count_nonzero(x[:,0]==0.0))
    print(x.shape)
    
    with h5py.File(new_hdf,'a') as f2:
        x_tmp = np.empty(x.shape)
        x.read_direct(x_tmp)
        f2.create_dataset(
            str('bub_X'),
            data = x_tmp,
            dtype   = np.float64,
            compression     = 'gzip',
            compression_opts= 9
        )
        f2.create_dataset(
            str('bub_Y'),
            (y.shape[0], N_z),
            dtype   = np.float64,
            compression     = 'gzip',
            compression_opts= 9
        )
        y2 = f2['bub_Y']
        y_tmp = np.empty(y.shape)
        y.read_direct(y_tmp)
        y2 = np.delete(y_tmp, [0,1,2], axis=1)

print("Ready")