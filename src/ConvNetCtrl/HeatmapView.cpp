#include "ConvNetCtrl.h"

namespace ConvNet {

HeatmapView::HeatmapView() {
	ses = NULL;
}

void HeatmapView::Paint(Draw& d) {
	Size sz = GetSize();
	if (!ses) {d.DrawRect(sz, White()); return;}
	
	
	ses->Enter();
	Net& net = ses->GetNetwork();
	
	int layer_count = net.GetLayers().GetCount();
	Font fnt = SansSerifZ(12);
	String fill_txt = "Value Function Approximating Neural Network:";
	Size txt_sz = GetTextSize(fill_txt, fnt);
	
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	id.DrawText(2, 2, fill_txt, fnt, Black());
	
	int x = 10;
	int y = 2 + txt_sz.cy + 2;
	int y3 = y + txt_sz.cy + 2;
	//double dx = (sz.cx - 50) / layer_count;
	//double dy = (sz.cy - 50) / layer_count;
	
	for(int i = 0; i < layer_count; i++) {
		LayerBase& lb = *net.GetLayers()[i];
		int output_length = lb.output_activation.GetLength();
		String layer_lbl = lb.GetKey() + "(" + IntStr(output_length) + ")";
		id.DrawText(x, y, layer_lbl, fnt, Black());
		int y2 = y3;
		tmp.SetCount(output_length);
		double max = 0;
		for(int j = 0; j < output_length; j++) {
			double d = lb.output_activation.Get(j);
			tmp[j] = d;
			if (d > 0) {
				if (d > max) max = d;
			}
			else {
				double neg = -d;
				if (neg > max) max = neg;
			}
		}
		for(int j = 0; j < output_length; j++) {
			double d = tmp[j] ;
			double v = fabs(d) / max * 255;
			if (d >= 0)	id.DrawRect(x, y2, 10, 10, Color(0,0,v));
			else		id.DrawRect(x, y2, 10, 10, Color(v,0,0));
			y2 += 12;
			if (y2 > sz.cy - 25) {
				y2 = y3;
				x += 12;
			}
		}
		x += 50;
	}
	
	ses->Leave();
	
	d.DrawImage(0, 0, id);
}

}
