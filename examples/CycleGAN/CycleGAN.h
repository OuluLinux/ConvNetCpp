#ifndef _CycleGAN_CycleGAN_h
#define _CycleGAN_CycleGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <CycleGAN/CycleGAN.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS CycleGANImg
#define IMAGEFILE <CycleGAN/CycleGAN.iml>
#include <Draw/iml_header.h>

// Forward declaration
class CycleGAN;

// Loss function types
enum class CycleGANLossType {
    BINARY_CROSS_ENTROPY,  // Standard GAN loss
    LEAST_SQUARES,         // LSGAN loss
    WASSERSTEIN            // WGAN loss
};

class CycleGANLayer {

protected:
	friend class CycleGAN;

	Session disc_X, disc_Y, gen_XtoY, gen_YtoX;  // Two discriminators and two generators
	OnlineAverage disc_X_cost_av, disc_Y_cost_av, gen_XtoY_cost_av, gen_YtoX_cost_av;
	Size sz;
	int input_width = 0, input_height = 0, input_depth = 0;
	int stride = 0;
	int data_iter = 0;
	int label = -1;
	CycleGANLossType loss_type = CycleGANLossType::BINARY_CROSS_ENTROPY;  // Default loss type
	double lambda_cycle = 10.0;  // Weight for cycle consistency loss

	// Temp
	CycleGAN* gan = NULL;
	Vector<double> tmp_ret, tmp_ret2;
	Volume tmp_input;

public:
	typedef CycleGANLayer CLASSNAME;

	CycleGANLayer();

	void Init(int stride);
	void SetLossType(CycleGANLossType type) { loss_type = type; }
	CycleGANLossType GetLossType() const { return loss_type; }
	void SetLambdaCycle(double lambda) { lambda_cycle = lambda; }
	double GetLambdaCycle() const { return lambda_cycle; }
	void Train();
	void SampleInput();
	void SampleOutput();
	Callback CallTrain() {return THISBACK(Train);}

	// Generate functions for both directions
	Volume& GenerateXtoY(Volume& input);
	Volume& GenerateYtoX(Volume& input);
	int GetStride() const {return stride;}
	Size GetSize() const {return Size(input_width, input_height);}

	Session& GetDiscriminatorX() {return disc_X;}
	Session& GetDiscriminatorY() {return disc_Y;}
	Session& GetGeneratorXtoY() {return gen_XtoY;}
	Session& GetGeneratorYtoX() {return gen_YtoX;}

	double PickAverageDiscriminatorXCost() {double d = disc_X_cost_av.mean; disc_X_cost_av.Clear(); return d;}
	double PickAverageDiscriminatorYCost() {double d = disc_Y_cost_av.mean; disc_Y_cost_av.Clear(); return d;}
	double PickAverageGeneratorXtoYCost() {double d = gen_XtoY_cost_av.mean; gen_XtoY_cost_av.Clear(); return d;}
	double PickAverageGeneratorYtoXCost() {double d = gen_YtoX_cost_av.mean; gen_YtoX_cost_av.Clear(); return d;}
};

class CycleGAN : public TopWindow {
	Splitter vsplit;
	WithCtrlPanel<ParentCtrl> panel;

	ConvNet::SessionConvLayers disc_X_layer_view, disc_Y_layer_view;
	ConvNet::SessionConvLayers gen_XtoY_layer_view, gen_YtoX_layer_view;
	Mutex lock;

	bool running = false, stopped = true;


public:
	typedef CycleGAN CLASSNAME;
	CycleGAN();
	~CycleGAN() {running = false; while (!stopped) Sleep(100);}

	void Init();

	void Training();

	void RefreshData();



	CycleGANLayer l;

};

#endif