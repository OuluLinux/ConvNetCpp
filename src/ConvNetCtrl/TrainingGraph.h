#ifndef _ConvNetCtrl_TrainingGraph_h_
#define _ConvNetCtrl_TrainingGraph_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>
#include <PlotCtrl/PlotCtrl.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class TrainingGraph : public ParentCtrl {
	
protected:
	friend class LayerView;
	
	Session* ses;
	PlotCtrl plotter;
	int average_size;
	int last_steps;
	int interval;
	int mode;
	
	enum {MODE_LOSS, MODE_REWARD};
	
public:
	typedef TrainingGraph CLASSNAME;
	TrainingGraph();
	
	void SetSession(Session& ses);
	void SetModeLoss() {mode = MODE_LOSS; plotter.data[0].SetTitle("Loss");}
	void SetModeReward() {mode = MODE_REWARD; plotter.data[0].SetTitle("Reward");}
	void SetAverage(int size) {average_size = size;}
	void SetInterval(int period) {interval = period;}
	
	void StepInterval(int num_steps);
	void RefreshData();
	void AddValue();
	void Clear();
};

}

#endif
