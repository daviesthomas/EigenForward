#include <model.h>
#include <Eigen/Eigen>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

MLP::MLP() {}

MLP::MLP(std::string fp) {
    this->load(fp);
}

void MLP::addLayer(Layer& l) {
    layers.push_back(l);
}

Eigen::VectorXf MLP::predict(Eigen::VectorXf& input) {
    Eigen::VectorXf activation = input;
            
    // forward pass signal through layers!
    for ( std::vector<Layer>::iterator it = layers.begin() ; it != layers.end(); ++it) {
        activation = it->forward(activation);
    }
    return activation;
}

bool MLP::load(std::string fp) {

    //validate fp exists
    HighFive::File file(fp, HighFive::File::ReadOnly);

    std::vector<std::string> layers = file.listObjectNames();

    for (std::vector<std::string>::iterator it = layers.begin() ; it != layers.end(); ++it) {
        // for each layer, copy weights to eigen
        HighFive::ObjectType objType = file.getObjectType(*it);

        if (objType != HighFive::ObjectType::Group) {
            std::cout << "Unspupported Layer\n";
            return 0;
        }

        HighFive::Group group = file.getGroup(*it);
        int n = group.getNumberObjects();
        
        if (n != 1) {
            std::cout << "Unsupported Layer\n";
            return 0;
        }

        group = group.getGroup(*it);
        std::vector<std::string> matNames = group.listObjectNames();

        Eigen::VectorXf bias;
        Eigen::MatrixXf weight;

        for (std::vector<std::string>::iterator matIt = matNames.begin(); matIt != matNames.end(); ++matIt) {
            objType = group.getObjectType(*matIt);
            if (objType != HighFive::ObjectType::Dataset) {
                std::cout << "Unsupported Layer\n";
                return 0;
            }

            // parse the weights and biases
            HighFive::DataSet dataset = group.getDataSet(*matIt);
            std::vector<size_t> dim = dataset.getDimensions();

            if (dim.size() == 1) {
                std::vector<float> bias_data;
                dataset.read(bias_data);
                // now to eigen!
                bias = Eigen::Map<Eigen::VectorXf>(bias_data.data(), bias_data.size());
            } else if (dim.size() == 2) {
                std::vector<std::vector<float>> weight_data;
                dataset.read(weight_data);
                weight.resize(dim[0], dim[1]);

                for (int i = 0; i < weight_data.size(); i ++) {
                    weight.row(i) = Eigen::Map<Eigen::VectorXf>(weight_data[i].data(), weight_data[i].size());
                }
            }
            else {
                std::cout << "Unsupported layer, to many dims!\n";
                return 0;
            }
        }

        Activations activation = Activations::Relu;

        if  ((it != layers.end()) && (next(it) == layers.end())) {
            activation = Activations::Tanh;
        } 

        Layer l(bias, weight, activation);
        this->addLayer(l);
    }


}
