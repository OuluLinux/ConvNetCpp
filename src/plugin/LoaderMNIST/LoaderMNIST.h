#ifndef _LoaderMNIST_LoaderMNIST_h
#define _LoaderMNIST_LoaderMNIST_h

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


class LoaderMNIST : public TopWindow {
	
	Label lbl;
	ProgressIndicator prog, sub;
	Button cancel;
	int ret_value;
	bool fast_cache;
	
public:
	typedef LoaderMNIST CLASSNAME;
	LoaderMNIST();
	
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
