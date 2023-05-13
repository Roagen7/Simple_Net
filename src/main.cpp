#include "Simple_NN.h"
#include <iostream>


bool is_in_rect(double x, double y){
    return x >= 3 && x <= 9 && y >= 2 && y <= 5;
}

train_set_t generate_points(uint32_t N){

    std::vector<Eigen::VectorXd> x_train;
    std::vector<Eigen::VectorXd> y_train;
    srand(1);

    for(int i = 0; i < N; i++){
        double x = ((double) rand()/RAND_MAX) * 10;
        double y = ((double) rand()/RAND_MAX) * 10;
        auto d = (double) is_in_rect(x,y);
        Eigen::VectorXd vxy(2,1);
        Eigen::VectorXd vd(1,1);
        vxy << x, y;
        vd << d;
        x_train.push_back(vxy);
        y_train.push_back(vd);
    }
    return {x_train, y_train};
}


int main() {
    Simple_NN model = Simple_NN::configure()
            .input_layer_size(2)
            .hidden_layer_sizes({4, 2})
            .output_layer_size(1)
            .verbose()
            .activation_fun(Simple_NN::RELU);

    train_set_t train_set = generate_points(10000);
    model.fit(train_set);

    Eigen::VectorXd x(2);
    x << 6, 3;
    Eigen::VectorXd x2(2);
    x2 << 1, 1;

    auto p = {x, x2};
    auto pred = model.predict(p);

}
