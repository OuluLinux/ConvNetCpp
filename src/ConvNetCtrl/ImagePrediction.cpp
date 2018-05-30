#include "ConvNetCtrl.h"

namespace ConvNet {

ImagePrediction::ImagePrediction() {
	max_count = 4 * 8;
	AddFrame(sb);
	sb.WhenScroll = THISBACK(Scroll);
	sb.SetLine(GetLineHeight());
	ses = NULL;
	augmentation = 0;
	do_flip = false;
}

void ImagePrediction::SetSession(Session& ses) {
	this->ses = &ses;
	ses.WhenStepInterval << THISBACK(StepInterval);
}

void ImagePrediction::StepInterval(int step_num) {
	
	// run prediction on test set
	if (step_num == 100 || (step_num % 1000) == 0) {
		RefreshData();
	}
	
}

void ImagePrediction::RefreshData() {
	if (!ses) return;
	
	lock.Enter();
	
	Net& net = ses->GetNetwork();
	SessionData& d = ses->Data();
	int layer_count = net.GetLayers().GetCount();
	ASSERT(layer_count);
	
	int num_classes = net.GetLayers()[layer_count-1].output_depth;
	
	int data_w = d.GetDataWidth();
	int data_h = d.GetDataHeight();
	int data_d = d.GetDataDepth();
	
	// grab a random test image
	int tests = 50;
	imgs.SetCount(tests);
	for (int num = 0; num < tests; num++) {
		
		int i = Random(d.GetDataCount());
		Vector<double>& vol = d.Get(i);
		int label = d.GetLabel(i);
		
		Image& img = imgs[num];
		ImageBuffer ib(data_w, data_h);
		RGBA* it = ib.Begin();
		for (int y = 0; y < data_h; y++) {
			for (int x = 0; x < data_w; x++) {
				if (data_d == 3) {
					it->r = Volume::Get(vol, x, y, 0, data_w, data_d) * 255;
					it->g = Volume::Get(vol, x, y, 1, data_w, data_d) * 255;
					it->b = Volume::Get(vol, x, y, 2, data_w, data_d) * 255;
				}
				else {
					byte b = Volume::Get(vol, x, y, 0, data_w, data_d) * 255;
					it->r = b;
					it->g = b;
					it->b = b;
				}
				it->a = 255;
				it++;
			}
		}
		img = ib;
		
		// forward prop it through the network
		aavg.Init(1, 1, num_classes, 0.0);
		
		// ensures we always have a list, regardless if above returns single item or list
		int n = 4;
		for (int i = 0; i < n; i++) {
			Volume aug(data_w, data_h, data_d, vol);
			if (augmentation)
				aug.Augment(augmentation, -1, -1, do_flip);
			Volume& a = net.Forward(aug);
			aavg.AddFrom(a);
		}
		
		double label_val = 0;
		VectorMap<int, double> preds;
		for(int i = 0; i < num_classes; i++) {
			double av = aavg.Get(i) / n;
			if (i == label)
				label_val = av;
			else
				preds.Add(i, av);
		}
		SortByValue(preds, StdGreater<double>());
		
		Add(img, d.GetClass(label), label_val, d.GetClass(preds.GetKey(0)), preds[0], d.GetClass(preds.GetKey(1)), preds[1]);
		
	}
	
	lock.Leave();
	
	PostCallback(THISBACK(Layout));
	PostCallback(THISBACK(Refresh0));
}

void ImagePrediction::Add(Image& img, String l0, double p0, String l1, double p1, String l2, double p2) {
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
