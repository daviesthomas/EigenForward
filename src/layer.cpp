#include <layer.h>
#include <iostream>

Layer::Layer(Eigen::VectorXf& biases, Eigen::MatrixXf& weights, Activations actType) {
    w = weights;
    b = biases;
    activation = actType;
}

Eigen::VectorXf Layer::forward(Eigen::VectorXf& input) {
    Eigen::VectorXf m = w.transpose() * input;
    m.colwise() += b;
    return this->activate(m); 
}

Eigen::VectorXf Layer::activate(Eigen::VectorXf& input) {
    if (activation == Activations::Tanh) {
        return input.array().tanh();
    } else if (activation == Activations::Relu) {
        return input.array().cwiseMax(float(0));
    } 
    std::cout << "No valid activation function...not activating!\n";
    return input;
}

void Layer::setActivation(Activations actType) {
    activation = actType;
}