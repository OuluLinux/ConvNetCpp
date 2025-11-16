#ifndef _BigGAN_BigGAN_h
#define _BigGAN_BigGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <BigGAN/BigGAN.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS BigGANImg
#define IMAGEFILE <BigGAN/BigGAN.iml>
#include <Draw/iml_header.h>

// Forward declaration
class BigGAN;

// Loss function types
enum class BigGANLossType {
    BINARY_CROSS_ENTROPY,  // Standard GAN loss
    LEAST_SQUARES,         // LSGAN loss
    WASSERSTEIN            // WGAN loss
};

class BigGANLayer {

protected:
	friend class BigGAN;

	Session disc, gen;
	OnlineAverage disc_cost_av, gen_cost_av;
	Size sz;
	int input_width = 0, input_height = 0, input_depth = 0;
	int noise_size = 128;   // Size of the noise vector
	int num_classes = 1000; // Number of classes for conditioning (e.g., ImageNet)
	int num_gen_layers = 6; // Number of layers in generator
	int num_disc_layers = 6; // Number of layers in discriminator
	double truncation_factor = 0.5; // For controlling sample variety
	int stride = 0;
	int data_iter = 0;
	int label = -1;
	BigGANLossType loss_type = BigGANLossType::BINARY_CROSS_ENTROPY;

	// Temp
	BigGAN* gan = NULL;
	Vector<double> tmp_ret, tmp_ret2;
	Volume tmp_input;

public:
	typedef BigGANLayer CLASSNAME;

	BigGANLayer();

	void Init(int stride, int noise_size = 128, int num_classes = 1000, int img_width = 128, int img_height = 128, int img_depth = 3);
	void SetLossType(BigGANLossType type) { loss_type = type; }
	BigGANLossType GetLossType() const { return loss_type; }
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

	// Set and get BigGAN-specific parameters
	void SetNoiseSize(int size) { noise_size = size; }
	int GetNoiseSize() const { return noise_size; }
	void SetNumClasses(int n) { num_classes = n; }
	int GetNumClasses() const { return num_classes; }
	void SetTruncationFactor(double factor) { truncation_factor = factor; }
	double GetTruncationFactor() const { return truncation_factor; }
};

class BigGAN : public TopWindow {
	Splitter vsplit;
	WithCtrlPanel<ParentCtrl> panel;

	ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
	Mutex lock;

	bool running = false, stopped = true;

public:
	typedef BigGAN CLASSNAME;
	BigGAN();
	~BigGAN() {running = false; while (!stopped) Sleep(100);}

	void Init();

	void Training();

	void RefreshData();

	BigGANLayer l;

};

#endif