#ifndef _ConvNetCtrl_BarView_h_
#define _ConvNetCtrl_BarView_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class BarView : public Ctrl {
	Session* ses;
	int space_between_bars;
	Color bar_clr;
	
public:
	typedef BarView CLASSNAME;
	BarView();
	
	virtual void Paint(Draw& d);
	
	void SetSession(Session& ses) {this->ses = &ses;}
	void SetColor(Color c) {bar_clr = c;}
	
};

}

#endif
