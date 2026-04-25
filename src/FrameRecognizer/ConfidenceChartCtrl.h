#ifndef _FrameRecognizer_ConfidenceChartCtrl_h_
#define _FrameRecognizer_ConfidenceChartCtrl_h_

#include <CtrlLib/CtrlLib.h>

NAMESPACE_UPP

class ConfidenceChartCtrl : public Ctrl {
public:
	typedef ConfidenceChartCtrl CLASSNAME;

	void AddSample(double confidence);
	void Clear();
	virtual void Paint(Draw& w) override;

private:
	Vector<double> samples_;
	static const int MAX_SAMPLES = 60;
};

END_UPP_NAMESPACE

#endif
