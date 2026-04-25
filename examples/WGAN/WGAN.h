#ifndef _WGAN_WGAN_h
#define _WGAN_WGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

// Forward declaration
class WGAN;

template <class T>
struct WithCtrlPanel : public T {
    TrainingGraph disc_graph;
    TrainingGraph gen_graph;

    void CtrlLayout(T& parent) {
        parent.Add(disc_graph.HSizePos().TopPos(0, 200));
        parent.Add(gen_graph.HSizePos().TopPos(200, 200));
    }
};

// WGAN specific parameters
struct WGANParams {
    double clip_value = 0.01;  // Weight clipping value for Lipschitz constraint
    int    critic_iter = 5;    // Number of critic iterations per generator iteration
};

class WGANLayer {

protected:
    friend class WGAN;

    Session disc, gen;
    OnlineAverage disc_cost_av, gen_cost_av;
    Size sz;
    int input_width = 0, input_height = 0, input_depth = 0;
    int stride = 0;
    int data_iter = 0;
    int label = -1;
    WGANParams wgan_params;

    // Temp
    WGAN* wgan = NULL;
    Vector<double> tmp_ret, tmp_ret2;
    Volume tmp_input;

public:
    typedef WGANLayer CLASSNAME;

    WGANLayer();

    void Init(int stride, const WGANParams& params = WGANParams());
    void SetParams(const WGANParams& params) { wgan_params = params; }
    WGANParams GetParams() const { return wgan_params; }
    void Train();
    void SampleInput();
    void SampleOutput();
    Callback CallTrain() {return THISBACK(Train);}

    Volume& Generate(Volume& input);
    int GetStride() const {return stride;}
    Size GetSize() const {return Size(input_width, input_height);}

    Session& GetDiscriminator() {return disc;}
    Session& GetGenerator() {return gen;}

    double PickAverageDiscriminatorCost() {double d = disc_cost_av.mean; disc_cost_av.Clear(); return d;}
    double PickAverageGeneratorCost() {double d = gen_cost_av.mean; gen_cost_av.Clear(); return d;}
};

class WGAN : public TopWindow {
    WithCtrlPanel<ParentCtrl> ctrl_panel;
    ParentCtrl options_panel;
    Splitter main_vsplit;

    ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
    Mutex lock;

    bool running = false, stopped = true;

public:
    typedef WGAN CLASSNAME;
    WGAN();
    ~WGAN() {running = false; while (!stopped) Sleep(100);}

    void Init();

    void Training();

    void RefreshData();

    WGANLayer l;
};

#endif
