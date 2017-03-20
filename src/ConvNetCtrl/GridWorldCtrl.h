#ifndef _ConvNetCtrl_GridWorldCtrl_h_
#define _ConvNetCtrl_GridWorldCtrl_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class GridWorldCtrl : public Ctrl {
	SpinLock		lock;
	Agent*			agent;
	int selected;
	
public:
	
	typedef GridWorldCtrl CLASSNAME;
	GridWorldCtrl();
	
	int GetSelected() const {return selected;}
	
	void SetAgent(Agent& agent) {this->agent = &agent;}
	
	virtual void LeftDown(Point p, dword keyflags);
	virtual void Paint(Draw& w);
	
	Callback WhenGridFocus, WhenGridUnfocus;
	
};

}

#endif
