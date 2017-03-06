#ifndef _ConvNetCtrl_ConvLayerCtrl_h_
#define _ConvNetCtrl_ConvLayerCtrl_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class ConvLayerCtrl : public Ctrl {
	Session* ses;
	VectorMap<int, Image> gradient_cache;
	int layer_id, height;
	bool hide_gradients;
	bool is_color;
	
public:
	typedef ConvLayerCtrl CLASSNAME;
	ConvLayerCtrl();
	
	int GetHeight() const {return height;}
	void DrawActivations(Draw& d, Size& sz, Point& pt, Volume& v, int scale, bool draw_grads=false, bool end_newline=true);
	void SetId(int i) {layer_id = i;}
	void SetSession(Session& ses) {this->ses = &ses;}
	void SetColor(bool b) {is_color = b;}
	void HideGradients(bool b=true) {hide_gradients = b;}
	void PaintSize(Draw& d, Size sz);
	void ClearGradientCache() {gradient_cache.Clear();}
	
	virtual void Paint(Draw& d);
	
};


class SessionConvLayers : public ParentCtrl {
	Array<ConvLayerCtrl> layer_ctrls;
	ScrollBar sb;
	Session* ses;
	bool is_scrolling;
	bool is_color;
	bool hide_gradients;
	
public:
	typedef SessionConvLayers CLASSNAME;
	SessionConvLayers();
	
	void SetSession(Session& ses);
	void SetColor(bool b=true) {is_color = b;}
	void Scroll();
	void RefreshLayers();
	void HideGradients(bool b=true) {hide_gradients = b;}
	virtual bool Key(dword key, int);
	virtual void MouseWheel(Point, int zdelta, dword);
	virtual void Layout();
	
};

}

#endif
