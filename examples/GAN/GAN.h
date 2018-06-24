#ifndef _GAN_GAN_h
#define _GAN_GAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include <plugin/png/png.h>

#define LAYOUTFILE <GAN/GAN.lay>
#include <CtrlCore/lay.h>
#define IMAGECLASS GANImg
#define IMAGEFILE <GAN/GAN.iml>
#include <Draw/iml_header.h>


class GANLayer {
	
protected:
	friend class GAN;
	
	Session disc, gen;
	OnlineAverage disc_cost_av, gen_cost_av;
	Size sz;
	int input_width = 0, input_height = 0, input_depth = 0;
	int stride = 0;
	int data_iter = 0;
	int label = -1;
	
	// Temp
	GAN* gan = NULL;
	Vector<double> tmp_ret, tmp_ret2;
	Volume tmp_input;
	
public:
	typedef GANLayer CLASSNAME;
	
	GANLayer();
	
	void Init(int stride);
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

class GAN : public TopWindow {
	Splitter vsplit;
	WithCtrlPanel<ParentCtrl> panel;
	
	ConvNet::SessionConvLayers disc_layer_view, gen_layer_view;
	Mutex lock;
	
	bool running = false, stopped = true;
	
	
public:
	typedef GAN CLASSNAME;
	GAN();
	~GAN() {running = false; while (!stopped) Sleep(100);}
	
	void Init();
	
	void Training();
	
	void RefreshData();
	
	
	
	GANLayer l;
	
};

#endif
