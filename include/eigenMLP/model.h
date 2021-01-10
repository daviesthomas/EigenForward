#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Eigen>
#include "layer.h"

class MLP {
    public:
        // base constructor
        MLP ();
        // load model constructor
        MLP (std::string fp);
        // load model from hdf5 
        bool load(std::string fp);
        // predict inference given input
        Eigen::VectorXf predict(Eigen::MatrixXf& input);
        // update existing layers activation function
        void updateLayerActivation(int idx, Activations act);
    private:
        std::vector<Layer> layers;
        void addLayer(Layer& l);
};

#endif