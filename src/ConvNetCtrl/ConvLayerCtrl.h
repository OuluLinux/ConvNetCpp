#ifndef _ConvNetCtrl_ConvLayerCtrl_h_
#define _ConvNetCtrl_ConvLayerCtrl_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class ConvLayerCtrl : public Ctrl {
	Session* ses;
	int layer_id, height;
	VectorMap<int, Image> gradient_cache;
	
public:
	typedef ConvLayerCtrl CLASSNAME;
	ConvLayerCtrl();
	
	int GetHeight() const {return height;}
	void DrawActivations(Draw& d, Size& sz, Point& pt, Volume& v, int scale, bool draw_grads=false, bool end_newline=true);
	void SetId(int i) {layer_id = i;}
	void SetSession(Session& ses) {this->ses = &ses;}
	void PaintSize(Draw& d, Size sz);
	void ClearGradientCache() {gradient_cache.Clear();}
	
	virtual void Paint(Draw& d);
	
};


class SessionConvLayers : public ParentCtrl {
	Array<ConvLayerCtrl> layer_ctrls;
	ScrollBar sb;
	Session* ses;
	bool is_scrolling;
	
public:
	typedef SessionConvLayers CLASSNAME;
	SessionConvLayers();
	
	void SetSession(Session& ses);
	void Scroll();
	void RefreshLayers();
	virtual bool Key(dword key, int);
	virtual void MouseWheel(Point, int zdelta, dword);
	virtual void Layout();
	
};

}

#endif
