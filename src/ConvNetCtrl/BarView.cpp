#include "ConvNetCtrl.h"

namespace ConvNet {

BarView::BarView() {
	ses = NULL;
	space_between_bars = 2;
	bar_clr = Black();
}

void BarView::Paint(Draw& d) {
	Size sz = GetSize();
	if (!ses) {d.DrawRect(sz, White()); return;}
	
	ses->Enter();
	
	#define ASSERTDRAW(x) if (!(x)) {ses->Leave(); d.DrawRect(sz, White()); return;}
	
	const Vector<double>& input = ses->GetLastInput();
	
	ASSERTDRAW(!input.IsEmpty());

	double density = (double) sz.cx / input.GetCount();
	ASSERTDRAW(density >= 1.0);
	int density_i = density + 0.5;
	if (density_i - space_between_bars > 0) density_i -= space_between_bars;
	
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	
	double x = 0;
	for(int i = 0; i < input.GetCount(); i++) {
		double out_act = input[i];
		int h = max(0.0, min(1.0, out_act)) * sz.cy;
		id.DrawRect((int)x, sz.cy - h, density_i, h, bar_clr);
		
		x += density_i + space_between_bars;
	}
	
	ses->Leave();
	
	d.DrawImage(0, 0, id);
}

}
