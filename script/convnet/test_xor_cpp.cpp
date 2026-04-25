// Standalone test: trains a net on XOR using ConvNet::Session directly.
// Does NOT require ByteVM, GUI, or plugins.
// Compile: clang++ -std=c++17 -I/common/active/sblo/Dev/ConvNetCpp/src \
//   test_xor_cpp.cpp /common/active/sblo/Dev/DS/bin/libConvNet.a -o test_xor

#include <ConvNet/ConvNet.h>
#include <cstdio>
#include <vector>

int main() {
    using namespace ConvNet;
    Session s;
    s.MakeLayers(
        "[{\"type\":\"input\",\"input_width\":1,\"input_height\":1,\"input_depth\":2},"
        "{\"type\":\"fc\",\"neuron_count\":6,\"activation\":\"relu\"},"
        "{\"type\":\"softmax\",\"class_count\":2}]"
    );
    s.GetTrainer().SetType(TRAINER_SGD);
    s.GetTrainer().SetLearningRate(0.01);
    s.GetTrainer().SetMomentum(0.9);
    s.GetTrainer().SetBatchSize(1);

    double xdata[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    int    ydata[4]    = {0, 1, 1, 0};

    Volume v;
    v.Init(2, 1, 1, 0.0);
    for(int epoch = 0; epoch < 300; epoch++) {
        for(int i = 0; i < 4; i++) {
            v.Set(0, xdata[i][0]);
            v.Set(1, xdata[i][1]);
            s.GetTrainer().Train(v, ydata[i], 1.0);
        }
    }

    std::vector<double> x01 = {0.0, 1.0};
    auto probs = s.Predict(x01);
    printf("XOR [0,1]: class0=%.3f class1=%.3f\n", probs[0], probs[1]);
    if(probs[1] > probs[0]) { printf("PASS\n"); return 0; }
    printf("FAIL\n");
    return 1;
}
