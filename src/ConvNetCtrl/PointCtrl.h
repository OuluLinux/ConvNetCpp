#ifndef _ConvNetCtrl_PointCtrl_h_
#define _ConvNetCtrl_PointCtrl_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class PointCtrl : public Ctrl {
	Session* ses;
	int vis_len, offset;
	
public:
	typedef PointCtrl CLASSNAME;
	PointCtrl();
	
	void RefreshData();
	void SetSession(Session& ses);
	void PostRefreshData() {PostCallback(THISBACK(RefreshData));}
	
	virtual void Paint(Draw& d);
	virtual void LeftDown(Point p, dword keyflags);
	
};

}

#endif
