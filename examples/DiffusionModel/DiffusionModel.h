#ifndef _DiffusionModel_DiffusionModel_h
#define _DiffusionModel_DiffusionModel_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <DiffusionModel/DiffusionModel.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS DiffusionModelImg
#define IMAGEFILE <DiffusionModel/DiffusionModel.iml>
#include <Draw/iml_header.h>

// Forward declaration
class DiffusionModel;

// Loss function types
enum class DiffusionModelLossType {
    MEAN_SQUARED_ERROR,    // Standard MSE loss
    L1_LOSS,               // L1 loss
    HUBER_LOSS             // Huber loss
};

class DiffusionModelLayer {

protected:
	friend class DiffusionModel;

	Session model;  // The main denoising model
	OnlineAverage loss_av;
	Size sz;
	int input_width = 0, input_height = 0, input_depth = 0;
	int timesteps = 1000;   // Number of diffusion steps
	Vector<double> betas;   // Noise schedule
	Vector<double> alphas;  // Complementary to betas
	Vector<double> alpha_bars;  // Cumulative alpha values
	int stride = 0;
	int data_iter = 0;
	int label = -1;
	DiffusionModelLossType loss_type = DiffusionModelLossType::MEAN_SQUARED_ERROR;

	// Temp
	DiffusionModel* dm = NULL;
	Vector<double> tmp_ret, tmp_ret2;
	Volume tmp_input;

public:
	typedef DiffusionModelLayer CLASSNAME;

	DiffusionModelLayer();

	void Init(int stride, int timesteps = 1000, int img_width = 32, int img_height = 32, int img_depth = 3);
	void SetLossType(DiffusionModelLossType type) { loss_type = type; }
	DiffusionModelLossType GetLossType() const { return loss_type; }
	void Train();
	void SampleInput();
	void SampleOutput();
	Callback CallTrain() {return THISBACK(Train);}

	// Generate function (denoise from random noise)
	Volume& Generate(int condition_label = -1); // -1 means no condition
	Volume& DenoiseStep(Volume& x, int t, int condition_label = -1);
	int GetStride() const {return stride;}
	Size GetSize() const {return Size(input_width, input_height);}

	Session& GetModel() {return model;}

	double PickAverageLoss() {double d = loss_av.mean; loss_av.Clear(); return d;}

	// Set and get Diffusion-specific parameters
	void SetTimesteps(int t) { timesteps = t; }
	int GetTimesteps() const { return timesteps; }
};

class DiffusionModel : public TopWindow {
	Splitter vsplit;
	WithCtrlPanel<ParentCtrl> panel;

	ConvNet::SessionConvLayers model_layer_view;
	Mutex lock;

	bool running = false, stopped = true;

public:
	typedef DiffusionModel CLASSNAME;
	DiffusionModel();
	~DiffusionModel() {running = false; while (!stopped) Sleep(100);}

	void Init();

	void Training();

	void RefreshData();

	DiffusionModelLayer l;

};

#endif