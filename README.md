# EigenForward
Simple header only dependency library for reading keras models and converting to Eigen for efficient CPU feed forward inference. 

Very limited for specific but will be extended over time.

# Dependencies

- Eigen3
- HighFive: https://github.com/BlueBrain/HighFive

# Build example app

mkdir build && cd build
cmake .. 
make

# TODO
  - Lol alot
  - Read keras .json file to init model (currently weights alone..)
  - Eigen Tensor refactor for batch dimension support (currently just a for loop)
  - Support skip connections/ not only simple mlp networks...
  - support convs! (this would be rad!) 


