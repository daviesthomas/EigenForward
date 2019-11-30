
import tensorflow as tf 
import numpy as np
from keras import backend as K
import h5py

f = h5py.File('armadillo_sdf.h5')
print(f.keys())

print(f['dense']['dense']['bias:0'])

