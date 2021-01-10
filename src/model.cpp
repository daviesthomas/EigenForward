#include <model.h>
#include <Eigen/Eigen>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

MLP::MLP() {}

MLP::MLP(std::string fp)
{
    bool ok = this->load(fp);

    if (!ok)
    {
        std::cerr << "Failed to load model!" << std::endl;
        //should cleanup...
    }
}

void MLP::addLayer(Layer &l)
{
    layers.push_back(l);
}

Eigen::VectorXf MLP::predict(Eigen::MatrixXf &input)
{
    Eigen::VectorXf result(input.rows());

    // This is not the way to handle batch dimension, should be factorized
    for (int i = 0; i < input.rows(); i++)
    {
        Eigen::VectorXf activation = input.row(i);
        for (std::vector<Layer>::iterator it = layers.begin(); it != layers.end(); ++it)
        {
            activation = it->forward(activation);
        }
        result.row(i) = activation;
    }
    return result;
}

void MLP::updateLayerActivation(int idx, Activations act)
{
    layers[idx].setActivation(act);
}

bool MLP::load(std::string fp)
{
    //validate fp exists
    HighFive::File file(fp, HighFive::File::ReadOnly);

    std::vector<std::string> layers = file.listObjectNames();

    if (layers.size() == 0)
    {
        std::cout << "Invalud H5 File... exiting\n";
        return false;
    }

    for (std::vector<std::string>::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        if ((*it).find("input") != std::string::npos)
        {
            // if input layer, just skip it... we don't support it.
            // however layer before this needs to have activation fixed if this input layer is final!!
            if ((it != layers.end()) && (next(it) == layers.end()))
            {
                std::cout <<"UPDATING TO TANH\n";
                updateLayerActivation(layers.size() - 2, Activations::Tanh);
            }
            continue;
        }

        // for each layer, copy weights to eigen
        HighFive::ObjectType objType = file.getObjectType(*it);

        if (objType != HighFive::ObjectType::Group)
        {
            std::cout << "Unspupported Layer... should not be group\n";
            return false;
        }

        HighFive::Group group = file.getGroup(*it);
        int n = group.getNumberObjects();

        if (n != 1)
        {
            std::cout << "Unsupported Layer... should only have one member\n";
            return false;
        }

        group = group.getGroup(*it);
        std::vector<std::string> matNames = group.listObjectNames();

        Eigen::VectorXf bias;
        Eigen::MatrixXf weight;

        for (std::vector<std::string>::iterator matIt = matNames.begin(); matIt != matNames.end(); ++matIt)
        {
            objType = group.getObjectType(*matIt);
            if (objType != HighFive::ObjectType::Dataset)
            {
                std::cout << "Unsupported Layer... should be dataset\n";
                return false;
            }

            // parse the weights and biases
            HighFive::DataSet dataset = group.getDataSet(*matIt);
            std::vector<size_t> dim = dataset.getDimensions();

            if (dim.size() == 1)
            {
                std::vector<float> bias_data;
                dataset.read(bias_data);
                // now to eigen!
                bias = Eigen::Map<Eigen::VectorXf>(bias_data.data(), bias_data.size());
            }
            else if (dim.size() == 2)
            {
                std::vector<std::vector<float>> weight_data;
                dataset.read(weight_data);
                weight.resize(dim[0], dim[1]);

                for (int i = 0; i < weight_data.size(); i++)
                {
                    weight.row(i) = Eigen::Map<Eigen::VectorXf>(weight_data[i].data(), weight_data[i].size());
                }
            }
            else
            {
                std::cout << "Unsupported layer, to many dims!\n";
                return false;
            }
        }

        Activations activation = Activations::Relu;

        if ((it != layers.end()) && (next(it) == layers.end()))
        {
            activation = Activations::Tanh;
        }

        Layer l(bias, weight, activation);
        this->addLayer(l);
    }
    return true;
}
