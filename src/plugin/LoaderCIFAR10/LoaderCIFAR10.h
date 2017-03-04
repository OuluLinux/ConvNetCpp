#ifndef _LoaderCIFAR10_LoaderCIFAR10_h
#define _LoaderCIFAR10_LoaderCIFAR10_h

#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;

class ImageBank {
	
	
public:
	
	Vector<int> labels, test_labels;
	Vector<Image> images, test_images;
	Vector<String> classes;
	
	void Serialize(Stream& s) {s % labels % test_labels % images % test_images % classes;}
	
};

inline ImageBank& GetImageBank() {return Single<ImageBank>();}


class LoaderCIFAR10 : public TopWindow {
	
	Label lbl;
	ProgressIndicator prog, sub;
	Button cancel;
	int ret_value;
	bool fast_cache;
	
public:
	typedef LoaderCIFAR10 CLASSNAME;
	LoaderCIFAR10();
	
	void Cancel() {ret_value = 1; Close();}
	void Load();
	void Progress(int actual, int total, String label);
	bool SubProgress(int actual, int total);
	void Close0() {Close();}
	bool IsFail() {return ret_value;}
	void SetFastCache(bool b=true) {fast_cache = b;}
	
};

}

#endif
