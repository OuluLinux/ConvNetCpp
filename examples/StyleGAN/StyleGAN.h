#ifndef _StyleGAN_StyleGAN_h
#define _StyleGAN_StyleGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <StyleGAN/StyleGAN.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS StyleGANImg
#define IMAGEFILE <StyleGAN/StyleGAN.iml>
#include <Draw/iml_header.h>

// Forward declaration
class StyleGAN;

// StyleGAN specific parameters
struct StyleGANParams {
    int latent_dim = 512;      // Dimension of the latent space (w-space in StyleGAN)
    int style_dim = 512;       // Dimension of the style vector
    int noise_dim = 4;         // Resolution of the noise input (typically 4x4)
    int initial_resolution = 4; // Starting resolution
    int target_resolution = 32;// Final resolution (keeping smaller for computational efficiency)
    double learning_rate = 0.0001;
};

class StyleGANLayer {

protected:
    friend class StyleGAN;

    Session disc, gen;
    OnlineAverage disc_cost_av, gen_cost_av;
    Size sz;
    int input_width = 0, input_height = 0, input_depth = 0;
    int stride = 0;
    int data_iter = 0;
    int label = -1;
    StyleGANParams stylegan_params;

    // Temp
    StyleGAN* stylegan = NULL;
    Vector<double> tmp_ret, tmp_ret2;
    Volume tmp_input;

public:
    typedef StyleGANLayer CLASSNAME;

    StyleGANLayer();

    void Init(int stride, const StyleGANParams& params = StyleGANParams());
    void SetParams(const StyleGANParams& params) { stylegan_params = params; }
    StyleGANParams GetParams() const { return stylegan_params; }
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

class StyleGAN : public TopWindow {
    Splitter vsplit;
    WithCtrlPanel<ParentCtrl> panel;

    ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
    Mutex lock;

    bool running = false, stopped = true;

public:
    typedef StyleGAN CLASSNAME;
    StyleGAN();
    ~StyleGAN() {running = false; while (!stopped) Sleep(100);}

    void Init();

    void Training();

    void RefreshData();

    StyleGANLayer l;
};

#endif