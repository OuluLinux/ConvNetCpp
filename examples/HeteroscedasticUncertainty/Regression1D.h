#ifndef _Regression1D_Regression1D_h
#define _Regression1D_Regression1D_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS Regression1DImg
#define IMAGEFILE <Regression1D/Regression1D.iml>
#include <Draw/iml_header.h>


class Regression1D : public DockWindow {
	
	HeatmapTimeView network_view;
	Button reload_btn;
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	
	LayerCtrl layer_ctrl;
	Label lbl_pointcount;
	EditIntSpin pointcount;
	Button regen;
	DropList funcs;
	
	double drop_prob = 0.05;
	
	Session ses;
	
	String t;
	int function;
	
public:
	typedef Regression1D CLASSNAME;
	Regression1D();
	
	virtual void DockInit();
	
	void Reload();
	void Regenerate();
	void Refresher();
	
};

#endif
