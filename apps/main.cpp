#define H5_USE_EIGEN

#include <iostream>

#include <Eigen/Eigen>

#include <model.h>
#include <layer.h>

const std::string FILE_NAME("../armadillo_sdf.h5");

int main()
{
    MLP mlp("../armadillo_sdf.h5"); 
    Eigen::VectorXf pointQuery(3); 
    pointQuery << 0.0,0.0,0.0;
    std::cout << mlp.predict(pointQuery) << std::endl;
}