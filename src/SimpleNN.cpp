#include "Simple_NN.h"
#include <algorithm>
#include <iostream>
#include <deque>

double relu(double x){
    return std::max(0., x);
}

double relu_derivative(double x){
    return x >= 0 ? 1 : 0;
}

Simple_NN_Builder Simple_NN::configure() {
    return {};
}

Simple_NN_Builder &Simple_NN_Builder::input_layer_size(const uint32_t& n) {
    nn.input_layer_size = n;
    return *this;
}

Simple_NN_Builder &Simple_NN_Builder::output_layer_size(const uint32_t& n) {
    nn.output_layer_size = n;
    return *this;
}

Simple_NN_Builder &Simple_NN_Builder::hidden_layer_sizes(const std::vector<uint32_t>& sizes) {
    nn.hidden_layer_sizes = sizes;
    return *this;
}

Simple_NN_Builder &Simple_NN_Builder::verbose() {
    nn.is_verbose = true;
    return *this;
}

Simple_NN_Builder &Simple_NN_Builder::activation_fun(Simple_NN::Activation f) {
    nn.activation_fun = f;
    return *this;
}

Simple_NN_Builder &Simple_NN_Builder::regression() {
    nn.net_type = Simple_NN::Net_Type::REGRESSION;
    return *this;
}

Simple_NN_Builder &Simple_NN_Builder::classification() {
    nn.net_type = Simple_NN::Net_Type::CLASSIFICATION;
    return *this;
}

void Simple_NN::initialize() {
    assert(input_layer_size != 0 && output_layer_size != 0);

    // weights initialization
    // remember bias (hence + 1)
    if(hidden_layer_sizes.empty()){
       weights.emplace_back(Eigen::MatrixXd::Random(output_layer_size, input_layer_size + 1));
    } else {
        weights.emplace_back(Eigen::MatrixXd::Random(hidden_layer_sizes[0], input_layer_size + 1));
    }

    for(size_t i = 0; i < hidden_layer_sizes.size()-1; i++){
        weights.emplace_back(Eigen::MatrixXd::Random(hidden_layer_sizes[i+1], hidden_layer_sizes[i] + 1));
    }

    if(!hidden_layer_sizes.empty()){
        weights.emplace_back(Eigen::MatrixXd::Random(output_layer_size, hidden_layer_sizes.back() + 1));
    }

    switch (activation_fun) {
        case RELU:
            activation_fptr = relu;
            activation_derivative_fptr = relu_derivative;
            break;
        case SIGMOID:
            assert(0);
        default:
            assert(0);
    }
}

Eigen::VectorXd Simple_NN::predict(const Eigen::VectorXd& x) {
    assert(x.rows() == input_layer_size);

    Eigen::VectorXd activation( x.rows() + 1, 1);
    activation << x, 1;

    for(const auto& layer_weights : weights){
        activation = layer_weights * activation;
        activation = activation.unaryExpr(activation_fptr);

        if(&layer_weights == &weights.back()) continue;
        activation.conservativeResize(activation.rows() + 1);
        activation(activation.rows()-1) = 1;
    }

    if(net_type == CLASSIFICATION && output_layer_size == 1){
        auto& cof = activation.coeffRef(0);
        if(cof >= 0.5){
            cof = 1;
        } else {
            cof = 0;
        }
    }

    return activation;
}

std::vector<Eigen::VectorXd> Simple_NN::predict(const std::vector<Eigen::VectorXd>& x) {
    std::vector<Eigen::VectorXd> predictions;
    predictions.reserve(x.size());
    for(const auto& xi : x){
        predictions.push_back(predict(xi));
    }
    return predictions;
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> Simple_NN::for_prop(const Eigen::VectorXd& x) {
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> nets_activations;
    Eigen::VectorXd activation( x.rows() + 1, 1);
    activation << x, 1;

    for(const auto& layer_weights : weights){
        activation = layer_weights * activation;
        nets_activations.first.push_back(activation);
        activation = activation.unaryExpr(activation_fptr);
        nets_activations.second.push_back(activation);
        if(&layer_weights == &weights.back()) continue;
        activation.conservativeResize(activation.rows() + 1);
        activation(activation.rows()-1) = 1;
    }

    return nets_activations;
}



void Simple_NN::fit(const dataset_t& train_set, const uint32_t& epochs) {
    const auto& x_train = train_set.first;
    const auto& y_train = train_set.second;

    for(uint32_t epoch = 0; epoch < epochs; epoch++){
        if(is_verbose){
            std::cout << "Epoch: " << epoch;
        }
        double mse = 0;

        for(size_t i = 0; i < x_train.size(); i++){
            mse += back_prop(x_train[i], y_train[i]);
        }

        if(is_verbose){
            std::cout << ", mse: " << mse << std::endl;
        }
    }
}



double Simple_NN::back_prop(const Eigen::VectorXd &x, const Eigen::VectorXd &d) {

    // forward pass
    auto outputs = for_prop(x); // activations and nets of layers (not input layer, since it's just x)
    const auto& nets = outputs.first;
    const auto& activations = outputs.second;
    double mse = 0;

    std::deque<Eigen::VectorXd> errs;

    Eigen::VectorXd err(d.rows());
    err << activations.back() - d;
    mse = err.squaredNorm()/err.rows();

    err = err.cwiseProduct(activations.back().unaryExpr(activation_derivative_fptr)); // maybe should be nets
    errs.push_front(err);
    for(int i = weights.size()-2; i >= 0; i--){
        const auto& wkp1 = weights[i+1];
        Eigen::VectorXd new_err(wkp1.rows());
        new_err = wkp1.transpose() * err;
        Eigen::VectorXd fz(nets[i].rows() + 1);
        fz << nets[i], 1;
        fz = fz.unaryExpr(activation_derivative_fptr);
        new_err = new_err.cwiseProduct(fz);
        err = new_err;
        err.conservativeResize(err.rows()-1); // get rid of the bias (don't back-propagate it)
        errs.push_front(err);
    }

    // update weights
    Eigen::VectorXd x_ext(x.rows() + 1);
    x_ext << x, 1;

    weights[0] -= learning_rate * errs[0] * x_ext.transpose();

    for(size_t i = 1; i < weights.size(); i++){
        Eigen::VectorXd a_ext(activations[i-1].rows() + 1);
        a_ext << activations[i-1], 1;

        weights[i] -= learning_rate * errs[i] * a_ext.transpose();
    }

    return mse;
}









