#ifndef _Classify2D_Classify2D_h_
#define _Classify2D_Classify2D_h_

#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS Classify2DImg
#define IMAGEFILE <Classify2D/Classify2D.iml>
#include <Draw/iml_header.h>


class Classify2D : public TopWindow {
	
protected:
	friend class PointCtrl;
	friend class LayerCtrl;
	
	Session session;
	bool running, stopped;
	String t;
	
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	Button reload_btn;
	Splitter v_split, h_split;
	ParentCtrl parent_pctrl;
	PointCtrl pctrl;
	Splitter pctrl_btns;
	Button btn_simple, btn_circle, btn_spiral, btn_random;
	LayerCtrl lctrl;
	
	
public:
	typedef Classify2D CLASSNAME;
	Classify2D();
	
	void ViewLayer(int i);
	void Reload();
	void RandomData(int count=40);
	void OriginalData();
	void CircleData(int count=100);
	void SpiralData(int count=100);
	void Refresher();
	
	void Start();
	
};


#endif
