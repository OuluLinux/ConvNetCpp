#ifndef _StyleGAN_h
#define _StyleGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>

using namespace Upp;
using namespace ConvNet;

struct StyleGANParams {
    int latent_dim = 512;
    int target_resolution = 32;
    double learning_rate = 0.0001;
};

class StyleGANLayer {
protected:
    friend class StyleGAN;
    Session disc, gen;
    OnlineAverage disc_cost_av, gen_cost_av;
    int input_width, input_height, input_depth;
    int stride;
    StyleGANParams stylegan_params;

public:
    StyleGANLayer();
    void Init(int stride, const StyleGANParams& params = StyleGANParams());
    void Train();
    void SampleInput();
    void SampleOutput();
    Volume& Generate(Volume& input);
    
    Session& GetDiscriminator() { return disc; }
    Session& GetGenerator() { return gen; }
    double PickAverageDiscriminatorCost() { double d = disc_cost_av.mean; disc_cost_av.Clear(); return d; }
    double PickAverageGeneratorCost() { double d = gen_cost_av.mean; gen_cost_av.Clear(); return d; }
};

class StyleGAN : public TopWindow {
    Splitter vsplit;
    ParentCtrl options_panel;
    TrainingGraph disc_graph, gen_graph;
    ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
    Mutex lock;
    bool running, stopped;
    StyleGANLayer l;

public:
    typedef StyleGAN CLASSNAME;
    StyleGAN();
    ~StyleGAN() { running = false; while (!stopped) Sleep(100); }
    void Init();
    void Training();
    void RefreshData();
};

#endif
