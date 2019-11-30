/*
Simple app showing the loading and initialization of MLP model

No input or output validation and model currently assumes MLP with 

relu hidden layers and tanh output... Sooo not terribly useful otherwise :)
*/


#define H5_USE_EIGEN

#include <iostream>
#include <Eigen/Eigen>
#include <model.h>
#include <layer.h>
#include <ctime>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Must supply path to h5 model from keras!\n";
        return 1;
    }
    
    MLP mlp(argv[1]); 
    Eigen::MatrixXf pointQueries(3,3); 
    //example of predicting 3 points
    pointQueries.row(0) = Eigen::Vector3f(0.1,0.2,0.1);
    pointQueries.row(1) = Eigen::Vector3f(-0.4,-0.2,0.2);
    pointQueries.row(2) = Eigen::Vector3f(0.0,0.0,0.0);

    
    std::clock_t startTime;
    startTime = std::clock();

    Eigen::VectorXf res;

    for (int i = 0; i < 10000; i ++ ) {
        // example prediction
        res = mlp.predict(pointQueries);
    }
    std::cout << res << std::endl;
    std::cout <<  "Took: " << (std::clock() - startTime)/(double)(CLOCKS_PER_SEC / 1000) << " ms to predict 30000 points...\n";
    

}