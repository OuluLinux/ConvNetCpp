#ifndef _LoaderMNIST_LoaderMNIST_h
#define _LoaderMNIST_LoaderMNIST_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>

namespace ConvNet {
using namespace Upp;

class LoaderMNIST : public TopWindow {
	SessionData* sd;
	Label lbl;
	ProgressIndicator prog, sub;
	Button cancel;
	int ret_value;
	
public:
	typedef LoaderMNIST CLASSNAME;
	LoaderMNIST(SessionData& sd);
	
	void Cancel() {ret_value = 1; Close();}
	void Load();
	void Progress(int actual, int total, String label);
	bool SubProgress(int actual, int total);
	void Close0() {Close();}
	bool IsFail() {return ret_value;}
	
};

}

#endif
