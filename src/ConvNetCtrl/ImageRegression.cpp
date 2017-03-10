#include "ConvNetCtrl.h"

namespace ConvNet {

ImageRegression::ImageRegression() {
	ses = NULL;
}
	
void ImageRegression::SetSession(Session& ses) {
	this->ses = &ses;
	ses.WhenStepInterval << THISBACK(StepInterval);
}

void ImageRegression::StepInterval(int step_num) {

	// run prediction on test set
	if (ts.Elapsed() > 2*1000) {
		RefreshData();
		ts.Reset();
	}
	
}

void ImageRegression::Paint(Draw& d) {
	Size sz = GetSize();
	
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	
	lock.Enter();
		
	int width = img_a.GetWidth();
	int height = img_a.GetHeight();
	
	int f_x = sz.cx / 2 - width - 10;
	int f_y = sz.cy / 2 - height / 2;
	id.DrawImage(f_x, f_y, img_a);
	
	int l_x = sz.cx / 2 + 10;
	int l_y = sz.cy / 2 - height / 2;
	id.DrawImage(l_x, l_y, img_b);
		
	lock.Leave();
	
	d.DrawImage(0,0,id);
}

void ImageRegression::RefreshData() {
	if (!ses) return;
	Session& ses = *this->ses;
	SessionData& d = ses.Data();
	
	Net& net = ses.GetNetwork();
	
	int pixel_count = d.GetDataCount();
	if (!pixel_count) return;
	
	tmp.SetCount(pixel_count*3);
	
	int width = img_a.GetWidth();
	int height = img_a.GetHeight();
	int depth = d.GetResult(0).GetCount();
	Volume in(1, 1, depth, 0);
	
	ses.Enter();
	
	int i = 0;
	for (int y = 0; y < height; y++) {
		
		in.Set(1, (double)y / height - 0.5);
		
		for (int x = 0; x < width; x++) {
			
			in.Set(0, (double)x / height - 0.5);
			
			Volume& out = net.Forward(in);
			
			if (depth == 3) {
				tmp[i++] = out.Get(0);
				tmp[i++] = out.Get(1);
				tmp[i++] = out.Get(2);
			} else {
				tmp[i++] = out.Get(0);
			}
		}
	}
	
	ses.Leave();
	
	ImageBuffer l_ib(width, height);
	RGBA* l_it = l_ib.Begin();
	
	i = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			
			if (depth == 3) {
				l_it->r = min(1.0, max(0.0, tmp[i++])) * 255;
				l_it->g = min(1.0, max(0.0, tmp[i++])) * 255;
				l_it->b = min(1.0, max(0.0, tmp[i++])) * 255;
			} else {
				byte b = min(1.0, max(0.0, tmp[i++])) * 255;
				l_it->r = b;
				l_it->g = b;
				l_it->b = b;
			}
			
			l_it->a = 255;
			l_it++;
		}
	}
	
	lock.Enter();
	img_b = l_ib;
	lock.Leave();
	
	PostCallback(THISBACK(Refresh));
}

}
