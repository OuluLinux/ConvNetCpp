#ifndef _LoaderMNIST_LoaderMNIST_h
#define _LoaderMNIST_LoaderMNIST_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>

namespace ConvNet {
using namespace Upp;

class LoaderMNIST : public TopWindow {
	Session* ses;
	Label lbl;
	ProgressIndicator prog, sub;
	Button cancel;
	int ret_value;
	
public:
	typedef LoaderMNIST CLASSNAME;
	LoaderMNIST(Session& ses);
	
	void Cancel() {ret_value = 1; Close();}
	void Load();
	void Progress(int actual, int total, String label);
	void SubProgress(int actual, int total);
	void Close0() {Close();}
	bool IsFail() {return ret_value;}
	
};

}

#endif
