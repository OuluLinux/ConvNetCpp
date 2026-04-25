#include <Core/Core.h>
#ifndef _ConditionalGAN_h
#define _ConditionalGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>

using namespace Upp;
using namespace ConvNet;

struct ConditionalGANParams {
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

class ConditionalGANLayer {
protected:
    friend class ConditionalGAN;
    Session disc, gen;
    OnlineAverage disc_cost_av, gen_cost_av;
    int input_width, input_height, input_depth;
    int stride;
    ConditionalGANParams conditionalgan_params;

public:
    ConditionalGANLayer();
    void Init(int stride, const ConditionalGANParams& params = ConditionalGANParams());
    void Train();
    void SampleInput();
    void SampleOutput();
    Volume& Generate(Volume& input);
    
    Session& GetDiscriminator() { return disc; }
    Session& GetGenerator() { return gen; }
    double PickAverageDiscriminatorCost() { double d = disc_cost_av.mean; disc_cost_av.Clear(); return d; }
    double PickAverageGeneratorCost() { double d = gen_cost_av.mean; gen_cost_av.Clear(); return d; }
};

class ConditionalGAN : public TopWindow {
    Splitter vsplit;
    ParentCtrl options_panel;
    WithCtrlPanel<ParentCtrl> ctrl_panel;
    ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
    Mutex lock;
    bool running, stopped;
    ConditionalGANLayer l;

public:
    typedef ConditionalGAN CLASSNAME;
    ConditionalGAN();
    ~ConditionalGAN() { running = false; while (!stopped) Sleep(100); }
    void Init();
    void Training();
    void RefreshData();
};

#endif
