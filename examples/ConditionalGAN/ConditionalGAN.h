#ifndef _ConditionalGAN_ConditionalGAN_h
#define _ConditionalGAN_ConditionalGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <ConditionalGAN/ConditionalGAN.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS ConditionalGANImg
#define IMAGEFILE <ConditionalGAN/ConditionalGAN.iml>
#include <Draw/iml_header.h>

// Forward declaration
class ConditionalGAN;

// Loss function types
enum class ConditionalGANLossType {
    BINARY_CROSS_ENTROPY,  // Standard GAN loss
    LEAST_SQUARES,         // LSGAN loss
    WASSERSTEIN            // WGAN loss
};

class ConditionalGANLayer {

protected:
	friend class ConditionalGAN;

	Session disc, gen;
	OnlineAverage disc_cost_av, gen_cost_av;
	Size sz;
	int input_width = 0, input_height = 0, input_depth = 0;
	int num_classes = 0;  // Number of classes for conditioning
	int noise_size = 0;   // Size of the noise vector
	int stride = 0;
	int data_iter = 0;
	int label = -1;
	ConditionalGANLossType loss_type = ConditionalGANLossType::BINARY_CROSS_ENTROPY;

	// Temp
	ConditionalGAN* gan = NULL;
	Vector<double> tmp_ret, tmp_ret2;
	Volume tmp_input;

public:
	typedef ConditionalGANLayer CLASSNAME;

	ConditionalGANLayer();

	void Init(int stride, int num_classes = 10, int noise_size = 100);
	void SetLossType(ConditionalGANLossType type) { loss_type = type; }
	ConditionalGANLossType GetLossType() const { return loss_type; }
	void Train();
	void SampleInput();
	void SampleOutput();
	Callback CallTrain() {return THISBACK(Train);}

	// Generate function with conditioning
	Volume& Generate(Volume& noise_input, int condition_label);
	int GetStride() const {return stride;}
	Size GetSize() const {return Size(input_width, input_height);}

	Session& GetDiscriminator() {return disc;}
	Session& GetGenerator() {return gen;}

	double PickAverageDiscriminatorCost() {double d = disc_cost_av.mean; disc_cost_av.Clear(); return d;}
	double PickAverageGeneratorCost() {double d = gen_cost_av.mean; gen_cost_av.Clear(); return d;}

	// Set and get conditioning parameters
	void SetNumClasses(int n) { num_classes = n; }
	int GetNumClasses() const { return num_classes; }
};

class ConditionalGAN : public TopWindow {
	Splitter vsplit;
	WithCtrlPanel<ParentCtrl> panel;

	ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
	Mutex lock;

	bool running = false, stopped = true;


public:
	typedef ConditionalGAN CLASSNAME;
	ConditionalGAN();
	~ConditionalGAN() {running = false; while (!stopped) Sleep(100);}

	void Init();

	void Training();

	void RefreshData();



	ConditionalGANLayer l;

};

#endif