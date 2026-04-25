#include "ConfidenceChartCtrl.h"

NAMESPACE_UPP

void ConfidenceChartCtrl::AddSample(double confidence)
{
	double v = minmax(confidence, 0.0, 1.0);
	samples_.Add(v);
	if(samples_.GetCount() > MAX_SAMPLES)
		samples_.Remove(0);
	Refresh();
}

void ConfidenceChartCtrl::Clear()
{
	samples_.Clear();
	Refresh();
}

void ConfidenceChartCtrl::Paint(Draw& w)
{
	Size sz = GetSize();
	w.DrawRect(sz, SColorFace());
	if(samples_.IsEmpty())
		return;

	int y50 = sz.cy - int(0.5 * sz.cy);
	w.DrawRect(0, y50, sz.cx, 1, Gray());

	int n = samples_.GetCount();
	int bw = max(1, sz.cx / MAX_SAMPLES);
	for(int i = 0; i < n; i++) {
		int h = int(samples_[i] * sz.cy);
		int x = sz.cx - (n - i) * bw;
		Color c = samples_[i] >= 0.5 ? Color(60, 180, 60) : Color(200, 80, 60);
		w.DrawRect(x, sz.cy - h, max(1, bw - 1), h, c);
	}
}

END_UPP_NAMESPACE
