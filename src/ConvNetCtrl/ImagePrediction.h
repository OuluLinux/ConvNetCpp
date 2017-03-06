#ifndef _ConvNetCtrl_ImagePrediction_h_
#define _ConvNetCtrl_ImagePrediction_h_

#include <ConvNet/ConvNet.h>
#include <CtrlLib/CtrlLib.h>

namespace ConvNet {
using namespace Upp;
using namespace ConvNet;

struct ImagePrediction : public Ctrl {
	SpinLock		lock;
	ScrollBar		sb;
	int				max_count;
	
	typedef Tuple3<double, String, bool> PredValue;
	
	struct Prediction : Moveable<Prediction> {
		Image* img;
		Vector<PredValue> values;
		bool operator() (const PredValue& a, const PredValue& b) const {return a.a > b.a;}
	};
	
	Array<Image> imgs;
	Vector<Prediction> preds;
	Session* ses;
	Volume aavg;
	int augmentation;
	bool do_flip;
	
protected:
	void Add(Image& img, String l0, double p0, String l1, double p1, String l2, double p2);
	void Refresh0() {Refresh();}
	
public:
	int GetLineHeight() { return 70; }

	virtual void Paint(Draw& w);
	virtual void Layout();
	virtual void MouseWheel(Point, int zdelta, dword);
	bool Key(dword key, int);
	void Scroll();
	void Clear() {lock.Enter(); preds.Clear(); lock.Leave();}
	void RefreshData();
	void SetSession(Session& ses);
	void StepInterval(int step_num);
	void SetAugmentation(int i, bool flip_img) {augmentation = i; do_flip = flip_img;}
	
	
	typedef ImagePrediction CLASSNAME;
	ImagePrediction();
};

}

#endif
