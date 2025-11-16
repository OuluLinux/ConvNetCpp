#ifndef _SAGAN_SAGAN_h
#define _SAGAN_SAGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <SAGAN/SAGAN.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS SAGANImg
#define IMAGEFILE <SAGAN/SAGAN.iml>
#include <Draw/iml_header.h>

// Forward declaration
class SAGAN;

// SAGAN specific parameters
struct SAGANParams {
    int attention_features = 16;  // Number of features to use for attention
    int latent_dim = 128;         // Dimension of the latent space
    int img_channels = 1;         // Number of image channels
    double learning_rate = 0.0002;
};

class SAGANLayer {

protected:
    friend class SAGAN;

    Session disc, gen;
    OnlineAverage disc_cost_av, gen_cost_av;
    Size sz;
    int input_width = 0, input_height = 0, input_depth = 0;
    int stride = 0;
    int data_iter = 0;
    int label = -1;
    SAGANParams sagan_params;

    // Temp
    SAGAN* sagan = NULL;
    Vector<double> tmp_ret, tmp_ret2;
    Volume tmp_input;

public:
    typedef SAGANLayer CLASSNAME;

    SAGANLayer();

    void Init(int stride, const SAGANParams& params = SAGANParams());
    void SetParams(const SAGANParams& params) { sagan_params = params; }
    SAGANParams GetParams() const { return sagan_params; }
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

class SAGAN : public TopWindow {
    Splitter vsplit;
    WithCtrlPanel<ParentCtrl> panel;

    ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
    Mutex lock;

    bool running = false, stopped = true;

public:
    typedef SAGAN CLASSNAME;
    SAGAN();
    ~SAGAN() {running = false; while (!stopped) Sleep(100);}

    void Init();

    void Training();

    void RefreshData();

    SAGANLayer l;
};

#endif