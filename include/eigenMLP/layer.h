#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Eigen>

enum Activations {
    Tanh, Relu
};

class Layer {
    public:
        Layer(Eigen::VectorXf& biases, Eigen::MatrixXf& weights, Activations actType);
        
        ~Layer(){}

        Eigen::VectorXf forward(Eigen::VectorXf& input);

    private:
        Eigen::MatrixXf w;
        Eigen::VectorXf b;
        int activation;

        Eigen::VectorXf activate(Eigen::VectorXf& input);
};

#endif