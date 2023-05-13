/*
 * Simple MLP
 * backpropagation algorithm:
 * http://cs229.stanford.edu/notes2020spring/cs229-notes-deep_learning.pdf
 * https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
 */

#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <Eigen/Dense>

using activation_fun_t = double (*)(double x);

using x_train_t = std::vector<Eigen::VectorXd>;
using y_train_t = std::vector<Eigen::VectorXd>;
using train_set_t = std::pair<x_train_t , y_train_t>;

class Simple_NN_Builder;

class Simple_NN {
public:
    enum Activation {
        RELU,
        SIGMOID
    };

    enum Optimizer {
        SGD,
        CROSS_ENTROPY
    };

    static Simple_NN_Builder configure();

    void fit(const train_set_t&, const uint32_t& epochs = 100);
    Eigen::VectorXd predict(const Eigen::VectorXd& x);
    std::vector<Eigen::VectorXd> predict(const std::vector<Eigen::VectorXd>& x);

private:
    friend class Simple_NN_Builder;

    bool is_verbose{};
    bool binary_output{};
    bool softmax_output{};
    uint32_t input_layer_size{};
    uint32_t output_layer_size{};
    std::vector<uint32_t> hidden_layer_sizes;
    std::vector<Eigen::MatrixXd> weights;
    activation_fun_t activation_fptr{nullptr};
    activation_fun_t activation_derivative_fptr{nullptr};
    Activation activation_fun{};
    Optimizer optimizer_fun{};
    double learning_rate{0.01};

    double back_prop(const Eigen::VectorXd& x, const Eigen::VectorXd& d);
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> for_prop(const Eigen::VectorXd& x); // returns nets and activations vector

    void initialize();

    Simple_NN() = default;

};


class Simple_NN_Builder {
public:
    operator Simple_NN&&(){
        nn.initialize();
        return std::move(nn);;
    }

    Simple_NN_Builder& input_layer_size(const uint32_t&);
    Simple_NN_Builder& output_layer_size(const uint32_t&);
    Simple_NN_Builder& hidden_layer_sizes(const std::vector<uint32_t>&);
    Simple_NN_Builder& verbose();
    Simple_NN_Builder& activation_fun(Simple_NN::Activation);
    Simple_NN_Builder& optimizer_fun(Simple_NN::Optimizer);

private:
    friend class Simple_NN;

    Simple_NN nn;
    Simple_NN_Builder() = default;

};

