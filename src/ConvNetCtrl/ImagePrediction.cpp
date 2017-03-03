#include "ConvNetCtrl.h"

namespace ConvNet {

ImagePrediction::ImagePrediction() {
	max_count = 4 * 8;
	AddFrame(sb);
	sb.WhenScroll = THISBACK(Scroll);
	sb.SetLine(GetLineHeight());
}

void ImagePrediction::Add(Image& img, String l0, double p0, String l1, double p1, String l2, double p2) {
	lock.Enter();
	while (preds.GetCount() >= max_count) preds.Remove(0);
	Prediction& pred = preds.Add();
	pred.img = &img;
	pred.values.SetCount(3);
	PredValue& v0 = pred.values[0];
	PredValue& v1 = pred.values[1];
	PredValue& v2 = pred.values[2];
	v0.a = max(0.0, min(1.0, p0));	v0.b = l0;	v0.c = true;
	v1.a = max(0.0, min(1.0, p1));	v1.b = l1;	v1.c = false;
	v2.a = max(0.0, min(1.0, p2));	v2.b = l2;	v2.c = false;
	Sort(pred.values, Prediction());
	lock.Leave();
	PostCallback(THISBACK(Layout));
}

void ImagePrediction::Paint(Draw& d) {
	Size sz = GetSize();
	
	ImageDraw id(sz);
	id.DrawRect(sz, SWhite);
	
	Font fnt = ArialZ(12);
	
	int h = GetLineHeight();
	int w = 260;
	int cols = max(1, sz.cx / w);
	
	int x = 0;
	int y1 = -sb;
	int y2 = y1 + h;
	
	lock.Enter();
	
	for(int i = 0; i < preds.GetCount(); i++) {
		Prediction& p = preds[i];
		
		if ((y2 > 0 && y2 < sz.cy) || (y1 < sz.cy && y1 > 0)) {
			
			id.DrawLine(0, y1, sz.cx, y1, 3, GrayColor(200));
			id.DrawLine(0, y2, sz.cx, y2, 3, GrayColor(100));
			id.DrawLine(x, y1, x, y2, 3, GrayColor(200));
			id.DrawLine(x+w, y1, x+w, y2, 3, GrayColor(200));
			
			Image rescaled = RescaleFilter(*p.img, h-8, h-8, FILTER_NEAREST);
			id.DrawImage(x+4, y1+4, rescaled);
			
			int sub_x = 4 + rescaled.GetSize().cx + 4;
			int sub_w = max(2, w - sub_x - 4);
			int sub_h = max(2, (h - 8) / p.values.GetCount());
			
			for(int j = 0; j < p.values.GetCount(); j++) {
				const PredValue& val = p.values[j];
				int sub_w2 = sub_w * val.a;
				int line_x = sub_x + x;
				int line_y = j * sub_h + 4 + y1;
				id.DrawRect(
					line_x,
					line_y,
					sub_w2,
					sub_h,
					val.c ? Color(85,187,85) : Color(187,85,85));
				id.DrawText(line_x, line_y, val.b, fnt);
			}
		}
		
		if ((i+1) % cols == 0) {
			y1 += h;
			y2 += h;
			x = 0;
		} else {
			x += w;
		}
	}
	
	lock.Leave();
	
	d.DrawImage(0, 0, id);
}

void ImagePrediction::Layout() {
	Size sz = GetSize();
	int h = GetLineHeight();
	int w = 260;
	int cols = max(1, sz.cx / w);
	int rows = preds.GetCount() / cols;
	if (preds.GetCount() % cols != 0) rows++;
	int total = rows * h;
	sb.SetTotal(total);
	sb.SetPage(sz.cy);
}

void ImagePrediction::MouseWheel(Point, int zdelta, dword) {
	sb.Wheel(zdelta);
}

bool ImagePrediction::Key(dword key, int) {
	return sb.VertKey(key);
}

void ImagePrediction::Scroll() {
	Refresh();
}

}
