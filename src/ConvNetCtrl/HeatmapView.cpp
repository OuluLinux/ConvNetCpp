#include "ConvNetCtrl.h"

namespace ConvNet {

HeatmapView::HeatmapView() {
	ses = NULL;
	graph = NULL;
	mode = -1;
}

void HeatmapView::Paint(Draw& d) {
	if (mode == MODE_SESSION)
		PaintSession(d);
	else if (mode == MODE_GRAPH)
		PaintGraph(d);
	else
		d.DrawRect(GetSize(), White());
}

void HeatmapView::PaintSession(Draw& d) {
	Size sz = GetSize();
	
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

void HeatmapView::PaintGraph(Draw& d) {
	Size sz = GetSize();
	
	int layer_count = graph->GetCount();
	Font fnt = SansSerifZ(12);
	String fill_txt = "Value Function Approximating Neural Network:";
	Size txt_sz = GetTextSize(fill_txt, fnt);
	
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	id.DrawText(2, 2, fill_txt, fnt, Black());
	
	int x = 10;
	int y = 2 + txt_sz.cy + 2;
	int y3 = y + txt_sz.cy + 2;
	
	for(int i = 1; i < layer_count; i++) {
		RecurrentBase& lb = graph->GetLayer(i);
		if (!lb.input1) continue;
		
		// Last volume in first layer is input, which might be invalid pointer
		int arg_count = lb.GetArgCount();
		int end = i == 0 ? arg_count-1 : arg_count;
		
		String layer_lbl = lb.GetKey() + "(";
		for(int j = 0; j < end; j++) {
			Volume& output = j == 0 ? *lb.input1 : *lb.input2;
			int output_length = output.GetLength();
			if (j) layer_lbl += ",";
			layer_lbl += IntStr(output_length);
		}
		layer_lbl += ")";
		id.DrawText(x, y, layer_lbl, fnt, Black());
		
		for(int j = 0; j < end; j++) {
			Volume& output = j == 0 ? *lb.input1 : *lb.input2;
			int output_length = output.GetLength();
			int y2 = y3;
			tmp.SetCount(output_length);
			double max = 0;
			for(int j = 0; j < output_length; j++) {
				double d = output.Get(j);
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
	}
	
	d.DrawImage(0, 0, id);
}


}
