#include <Core/Core.h>
#ifndef _BigGAN_h
#define _BigGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>

using namespace Upp;
using namespace ConvNet;

struct BigGANParams {
    double learning_rate = 0.0001;
    int latent_dim = 100;
    int target_resolution = 32;
};

template <class T>
struct WithCtrlPanel : public T {
    TrainingGraph disc_graph;
    TrainingGraph gen_graph;

    void CtrlLayout(T& parent) {
        parent.Add(disc_graph.HSizePos().TopPos(0, 200));
        parent.Add(gen_graph.HSizePos().TopPos(200, 200));
    }
};

class BigGANLayer {
protected:
    friend class BigGAN;
    Session disc, gen;
    OnlineAverage disc_cost_av, gen_cost_av;
    int input_width, input_height, input_depth;
    int stride;
    BigGANParams biggan_params;

public:
    BigGANLayer();
    void Init(int stride, const BigGANParams& params = BigGANParams());
    void Train();
    void SampleInput();
    void SampleOutput();
    Volume& Generate(Volume& input);
    
    Session& GetDiscriminator() { return disc; }
    Session& GetGenerator() { return gen; }
    double PickAverageDiscriminatorCost() { double d = disc_cost_av.mean; disc_cost_av.Clear(); return d; }
    double PickAverageGeneratorCost() { double d = gen_cost_av.mean; gen_cost_av.Clear(); return d; }
};

class BigGAN : public TopWindow {
    Splitter vsplit;
    ParentCtrl options_panel;
    WithCtrlPanel<ParentCtrl> ctrl_panel;
    ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
    Mutex lock;
    bool running, stopped;
    BigGANLayer l;

public:
    typedef BigGAN CLASSNAME;
    BigGAN();
    ~BigGAN() { running = false; while (!stopped) Sleep(100); }
    void Init();
    void Training();
    void RefreshData();
};

#endif
