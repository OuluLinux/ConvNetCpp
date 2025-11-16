#ifndef _ProgressiveGAN_ProgressiveGAN_h
#define _ProgressiveGAN_ProgressiveGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <ProgressiveGAN/ProgressiveGAN.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS ProgressiveGANImg
#define IMAGEFILE <ProgressiveGAN/ProgressiveGAN.iml>
#include <Draw/iml_header.h>

// Forward declaration
class ProgressiveGAN;

// Progressive Growing GAN specific parameters
struct ProgressiveGANParams {
    int initial_resolution = 4;  // Starting resolution (4x4)
    int target_resolution = 128; // Final resolution to grow to
    double resolution_growth_interval = 10000; // Steps to double resolution
    double fade_in_duration = 0.5; // Fraction of steps to fade in new layers
};

class ProgressiveGANLayer {

protected:
    friend class ProgressiveGAN;

    Session disc, gen;
    OnlineAverage disc_cost_av, gen_cost_av;
    Size sz;
    int input_width = 0, input_height = 0, input_depth = 0;
    int stride = 0;
    int data_iter = 0;
    int label = -1;
    ProgressiveGANParams prog_params;
    
    // Current resolution in the progressive growing process
    int current_resolution = 4;
    int current_phase_step = 0;  // Step within current phase
    int phase_steps = 10000;     // Total steps per resolution phase

    // Temp
    ProgressiveGAN* prog_gan = NULL;
    Vector<double> tmp_ret, tmp_ret2;
    Volume tmp_input;

public:
    typedef ProgressiveGANLayer CLASSNAME;

    ProgressiveGANLayer();

    void Init(int stride, const ProgressiveGANParams& params = ProgressiveGANParams());
    void SetParams(const ProgressiveGANParams& params) { prog_params = params; }
    ProgressiveGANParams GetParams() const { return prog_params; }
    void Train();
    void SampleInput();
    void SampleOutput();
    Callback CallTrain() {return THISBACK(Train);}

    Volume& Generate(Volume& input);
    int GetStride() const {return stride;}
    Size GetSize() const {return Size(input_width, input_height);}
    
    // Methods for progressive growing
    void UpdateResolution();
    void AddResolutionBlock();
    double GetAlpha(); // For fading in new layers

    Session& GetDiscriminator() {return disc;}
    Session& GetGenerator() {return gen;}

    double PickAverageDiscriminatorCost() {double d = disc_cost_av.mean; disc_cost_av.Clear(); return d;}
    double PickAverageGeneratorCost() {double d = gen_cost_av.mean; gen_cost_av.Clear(); return d;}
};

class ProgressiveGAN : public TopWindow {
    Splitter vsplit;
    WithCtrlPanel<ParentCtrl> panel;

    ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
    Mutex lock;

    bool running = false, stopped = true;

public:
    typedef ProgressiveGAN CLASSNAME;
    ProgressiveGAN();
    ~ProgressiveGAN() {running = false; while (!stopped) Sleep(100);}

    void Init();

    void Training();

    void RefreshData();

    ProgressiveGANLayer l;
};

#endif