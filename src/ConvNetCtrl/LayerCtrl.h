#ifndef _ConvNetCtrl_LayerCtrl_h_
#define _ConvNetCtrl_LayerCtrl_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class LayerCtrl;

class LayerView : public Ctrl {
	typedef Tuple2<Point, Point> Line;
	Vector<Vector<Point> > lines;
	Vector<double> gridx;
	Vector<double> gridy;
	Vector<bool> gridl;
	Vector<Pointf> tmp_pts;
	Vector<Point> tmp_pts1;
	Vector<Vector<Point> > tmp_pts2;
	Vector<Image> tmp_imgs;
	Vector<Vector<double> > volumes;
	Vector<Color> lbl_colors;
	Vector<int> labels;
	LayerCtrl* lc;
	
	// tmp vars
	int x_off, y_off, count, lix, d0, d1, vis_len, density, density_2;
	
	VectorMap<double, Window> sum_y, sum_y_sq, sum_sigma2;
	Vector<Point> polygon;
	
	void PaintInputX(Draw& d);
	void PaintInputXY(Draw& d);
	void PaintInputImage(Draw& d);
	
public:
	LayerView(LayerCtrl* lc);
	
	virtual void Paint(Draw& d);
	void ClearCache();
	
};


class LayerCtrl : public ParentCtrl {
	
protected:
	friend class LayerView;
	
	Session* ses;
	int d0, d1;
	int lix;
	
	Array<ButtonOption> layer_buttons;
	
	LayerView view;
	Button btn_cycle;
	Label lbl_layer;
	Splitter layerbtn_split;
	
	void RefreshCycle();
	
public:
	typedef LayerCtrl CLASSNAME;
	LayerCtrl();
	
	void SetSession(Session& ses);
	
	void ViewLayer(int i);
	void Cycle();
	void RefreshData();
	void PostRefreshData() {PostCallback(THISBACK(RefreshData));}
	int GetId() const {return lix;}
	
};

}

#endif
