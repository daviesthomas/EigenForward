#include <Eigen/Eigen>

enum Activations {
    Tanh, Relu
};

class Layer {
    public:
        Layer(Eigen::VectorXf& biases, Eigen::MatrixXf& weights, Activations actType) {
            w = weights;
            b = biases;
            activation = actType;
        }

        ~Layer(){}

        Eigen::VectorXf forward(Eigen::VectorXf& input) {
            Eigen::VectorXf m = w.transpose() * input;
            m.colwise() += b;

            return this->Activation(m);  
        }
        
    private:
        Eigen::MatrixXf w;
        Eigen::VectorXf b;
        int activation;

        Eigen::VectorXf Activation(Eigen::VectorXf& input) {
            if (activation == Activations::Tanh) {
                std::cout <<"TANH\n";
                return input.array().tanh();
            } else if (activation == Activations::Relu) {
                return input.array().cwiseMax(float(0));
            } 
            std::cout << "No valid activation function...not activating!\n";
            return input;
        }
};
 

class MLP {
    public:
        MLP () {}

        void addLayer(Layer& l) {
            layers.push_back(l);
        }

        Eigen::VectorXf predict(Eigen::VectorXf& input) {
            Eigen::VectorXf activation = input;
            
            // forward pass signal through layers!
            for ( std::vector<Layer>::iterator it = layers.begin() ; it != layers.end(); ++it) {
                activation = it->forward(activation);
            }
            return activation;
        }

    private:
        std::vector<Layer> layers;
};


