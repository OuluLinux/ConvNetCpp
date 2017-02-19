#ifndef _ConvNetCtrl_PointCtrl_h_
#define _ConvNetCtrl_PointCtrl_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>
using namespace Upp;

namespace ConvNetCtrl {
using namespace ConvNet;

class PointCtrl : public Ctrl {
	Session* ses;
	int vis_len, offset;
	bool sync_trainer;
	
public:
	typedef PointCtrl CLASSNAME;
	PointCtrl(Session& ses);
	
	void RefreshData();
	
	virtual void Paint(Draw& d);
	virtual void LeftDown(Point p, dword keyflags);
	
	void SetSyncTrainer(bool b=false) {sync_trainer = b;}
	
};

}

#endif
