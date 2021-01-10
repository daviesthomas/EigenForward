
import tensorflow as tf 
import numpy as np
from keras import backend as K
import h5py

f0 = h5py.File('armadillo_sdf.h5')
f1 = h5py.File('1887176.h5')
f2 = h5py.File('100026_sf.h5')


print(f0.keys(), f0.name)
print(f0['/'])
print(f1.keys(), f1.name)
print(f2.keys(), f2.name)


print(f1['dense']['dense']['bias:0'])