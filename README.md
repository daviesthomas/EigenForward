# EigenForward
Simple header only dependency library for reading keras models and converting to Eigen for efficient CPU feed forward inference. 

Very limited for specific but will be extended over time.

# Dependencies

- Eigen3
- HighFive: https://github.com/BlueBrain/HighFive

on osx we need
``` brew install hdf5 ```
``` brew install boost ```

# Build example app

mkdir build && cd build
cmake .. 
make



