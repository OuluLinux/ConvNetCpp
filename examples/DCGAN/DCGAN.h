#ifndef _DCGAN_DCGAN_h
#define _DCGAN_DCGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <DCGAN/DCGAN.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS DCGANImg
#define IMAGEFILE <DCGAN/DCGAN.iml>
#include <Draw/iml_header.h>

// Forward declaration
class DCGAN;

// Loss function types
enum class GANLossType {
    BINARY_CROSS_ENTROPY,  // Standard GAN loss
    LEAST_SQUARES,         // LSGAN loss
    WASSERSTEIN            // WGAN loss
};

class DCGANLayer {

protected:
    friend class DCGAN;

    Session disc, gen;
    OnlineAverage disc_cost_av, gen_cost_av;
    Size sz;
    int input_width = 0, input_height = 0, input_depth = 0;
    int stride = 0;
    int data_iter = 0;
    int label = -1;
    GANLossType loss_type = GANLossType::BINARY_CROSS_ENTROPY;  // Default loss type

    // Temp
    DCGAN* dcgan = NULL;
    Vector<double> tmp_ret, tmp_ret2;
    Volume tmp_input;

public:
    typedef DCGANLayer CLASSNAME;

    DCGANLayer();

    void Init(int stride);
    void SetLossType(GANLossType type) { loss_type = type; }
    GANLossType GetLossType() const { return loss_type; }
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

class DCGAN : public TopWindow {
    Splitter vsplit;
    WithCtrlPanel<ParentCtrl> panel;

    ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
    Mutex lock;

    bool running = false, stopped = true;

public:
    typedef DCGAN CLASSNAME;
    DCGAN();
    ~DCGAN() {running = false; while (!stopped) Sleep(100);}

    void Init();

    void Training();

    void RefreshData();

    DCGANLayer l;
};

#endif