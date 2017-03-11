#ifndef _ConvNetCtrl_MetaSessionGraph_h_
#define _ConvNetCtrl_MetaSessionGraph_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>
#include <PlotCtrl/PlotCtrl.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class MetaSessionGraph : public ParentCtrl {
	
protected:
	friend class LayerView;
	
	String title;
	MetaSession* mses;
	PlotCtrl plot;
	LegendCtrl legend;
	double max, min;
	int last_steps;
	int interval;
	int mode;
	bool show_legend;
	
	enum {MODE_LATESTLOSS, MODE_LOSS, MODE_TRAINACC, MODE_TESTACC};
	
	void ResizeLegend();
	
public:
	typedef MetaSessionGraph CLASSNAME;
	MetaSessionGraph();
	
	void SetMetaSession(MetaSession& mses);
	void SetModeLatestLoss()		{mode = MODE_LATESTLOSS;	title = "Latest Loss";}
	void SetModeLoss()				{mode = MODE_LOSS;			title = "Loss";}
	void SetModeTrainingAccuracy()	{mode = MODE_TRAINACC;		title = "Training accuracy";}
	void SetModeTestAccuracy()		{mode = MODE_TESTACC;		title = "Test accuracy";}
	void SetInterval(int period)	{interval = period;}
	
	void StepInterval(int num_steps);
	void RefreshData();
	void AddValue();
	void Clear();
	void PostClear() {PostCallback(THISBACK(Clear));}
	void HideLegend(bool b=true) {show_legend = !b;}
	
};

}

#endif
