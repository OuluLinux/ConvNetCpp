#ifndef _ConvNetCtrl_LayerCtrl_h_
#define _ConvNetCtrl_LayerCtrl_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>
using namespace Upp;

namespace ConvNetCtrl {
using namespace ConvNet;

class LayerCtrl;

class LayerView : public Ctrl {
	typedef Tuple2<Point, Point> Line;
	Vector<Vector<Point> > lines;
	Vector<double> gridx;
	Vector<double> gridy;
	Vector<bool> gridl;
	LayerCtrl* lc;
	
public:
	LayerView(LayerCtrl* lc);
	
	virtual void Paint(Draw& d);
	
};


class LayerCtrl : public ParentCtrl {
	
protected:
	friend class LayerView;
	
	Session* ses;
	int d0, d1;
	int lix;
	bool sync_trainer;
	
	Array<Button> layer_buttons;
	
	LayerView view;
	Button btn_cycle;
	Label lbl_layer;
	Splitter layerbtn_split;
	
	void RefreshCycle();
	
public:
	typedef LayerCtrl CLASSNAME;
	LayerCtrl(Session& ses);
	
	void ViewLayer(int i);
	void Cycle();
	void RefreshData();
	void PostRefreshData() {PostCallback(THISBACK(RefreshData));}
	int GetId() const {return lix;}
	void SetSyncTrainer(bool b=false) {sync_trainer = b;}
	
};

}

#endif
