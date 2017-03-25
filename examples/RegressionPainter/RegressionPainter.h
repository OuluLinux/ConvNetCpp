#ifndef _RegressionPainter_RegressionPainter_h
#define _RegressionPainter_RegressionPainter_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <Docking/Docking.h>
using namespace ConvNet;
using namespace Upp;

#define IMAGECLASS RegressionPainterImg
#define IMAGEFILE <RegressionPainter/RegressionPainter.iml>
#include <Draw/iml_header.h>


struct PainterImage : public Display {
	virtual void Paint(Draw& w, const Rect& r, const Value& q,
	                   Color ink, Color paper, dword style) const {
		w.DrawRect(r, paper);
		Image img;
		int i = q;
		switch (i) {
			case 0: img = RegressionPainterImg::cat; break;
			case 1: img = RegressionPainterImg::battery; break;
			case 2: img = RegressionPainterImg::chess; break;
			case 3: img = RegressionPainterImg::chip; break;
			case 4: img = RegressionPainterImg::dora; break;
			case 5: img = RegressionPainterImg::earth; break;
			case 6: img = RegressionPainterImg::esher; break;
			case 7: img = RegressionPainterImg::fox; break;
			case 8: img = RegressionPainterImg::fractal; break;
			case 9: img = RegressionPainterImg::gradient; break;
			case 10: img = RegressionPainterImg::jitendra; break;
			case 11: img = RegressionPainterImg::pencils; break;
			case 12: img = RegressionPainterImg::rainforest; break;
			case 13: img = RegressionPainterImg::reddit; break;
			case 14: img = RegressionPainterImg::rubiks; break;
			case 15: img = RegressionPainterImg::starry; break;
			case 16: img = RegressionPainterImg::tesla; break;
			case 17: img = RegressionPainterImg::twitter; break;
			case 18: img = RegressionPainterImg::usa; break;
		}
		Size sz = img.GetSize();
		sz /=  (double)sz.cy / r.GetHeight();
		
		img = Rescale(img, sz);
		w.DrawImage(r.left + r.Width() / 2 - sz.cx / 2, r.top, img);
	}
};


class RegressionPainter : public DockWindow {
	Session ses;
	String t;
	
	ArrayCtrl img_list;
	ConvNet::ImageRegression img_ctrl;
	
	Button reload_btn;
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	Label status;
	
	ParentCtrl slider_ctrl;
	SliderCtrl slider;
	Label lbl_slider, rate_info;
	
public:
	typedef RegressionPainter CLASSNAME;
	RegressionPainter();
	~RegressionPainter() {ses.StopTraining();}
	
	virtual void DockInit();
	
	void SetSlider(int i) {slider.SetData(i); RefreshLearningRate();}
	void StepInterval(int i);
	void RefreshStatus();
	void RefreshLearningRate();
	void SetImage();
	void Refresher();
	void Reload();
	
};

#endif
