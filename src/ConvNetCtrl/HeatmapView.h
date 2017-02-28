#ifndef _ConvNetCtrl_HeatmapView_h_
#define _ConvNetCtrl_HeatmapView_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class HeatmapView : public Ctrl {
	Session* ses;
	Vector<double> tmp;
	
public:
	typedef HeatmapView CLASSNAME;
	HeatmapView();
	
	virtual void Paint(Draw& d);
	
	void SetSession(Session& ses) {this->ses = &ses;}
	
};

}

#endif
