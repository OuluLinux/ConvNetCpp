#ifndef _ConvNetCtrl_ImageRegression_h_
#define _ConvNetCtrl_ImageRegression_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

class ImageRegression : public Ctrl {
	Session* ses;
	Image img_a, img_b;
	SpinLock lock;
	Vector<double> tmp;
	TimeStop ts;
	
public:
	typedef ImageRegression CLASSNAME;
	ImageRegression();
	
	void SetSource(const Image& img) {img_a = img; img_b.Clear(); ts.Reset(); PostCallback(THISBACK(Refresh));}
	void SetSession(Session& ses);
	void RefreshData();
	void StartRefreshData() {Thread::Start(THISBACK(RefreshData));}
	void Refresh() {Ctrl::Refresh();}
	void StepInterval(int step_num);
	
	virtual void Paint(Draw& d);
	
};

}

#endif
