#ifndef _LoaderCIFAR10_LoaderCIFAR10_h
#define _LoaderCIFAR10_LoaderCIFAR10_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>

namespace ConvNet {
using namespace Upp;

class LoaderCIFAR10 : public TopWindow {
	Session* ses;
	Label lbl;
	ProgressIndicator prog, sub;
	Button cancel;
	int ret_value;
	
public:
	typedef LoaderCIFAR10 CLASSNAME;
	LoaderCIFAR10(Session& ses);
	
	void Cancel() {ret_value = 1; Close();}
	void Load();
	void Progress(int actual, int total, String label);
	bool SubProgress(int actual, int total);
	void Close0() {Close();}
	bool IsFail() {return ret_value;}
	
};

}

#endif
