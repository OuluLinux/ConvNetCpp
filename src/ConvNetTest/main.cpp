#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN {
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

    Volume v;
    v.Init(2, 1, 1, 0.0);
    for(int epoch = 0; epoch < 300; epoch++) {
        for(int i = 0; i < 4; i++) {
            static double xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
            static int    yd[4]    = {0, 1, 1, 0};
            v.Set(0, xd[i][0]);
            v.Set(1, xd[i][1]);
            s.GetTrainer().Train(v, yd[i], 1.0);
        }
    }

    Vector<double> x01;
    x01.Add(0.0);
    x01.Add(1.0);
    Vector<double> probs = s.Predict(x01);
    Cout() << "XOR [0,1]: class0=" << probs[0] << " class1=" << probs[1] << "\n";
    if(probs.GetCount() >= 2 && probs[1] > probs[0]) {
        Cout() << "PASS\n";
        SetExitCode(0);
    }
    else {
        Cout() << "FAIL\n";
        SetExitCode(1);
    }
}
