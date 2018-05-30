#include "ConvNetCtrl.h"

namespace ConvNet {

PointCtrl::PointCtrl() {
	vis_len = 0;
	offset = 0;
	ses = NULL;
}

void PointCtrl::SetSession(Session& ses) {
	this->ses = &ses;
	ses.WhenSessionLoaded << THISBACK(PostRefreshData);
}

void PointCtrl::RefreshData() {
	Refresh();
}

void PointCtrl::Paint(Draw& draw) {
	if (!ses) {draw.DrawRect(GetSize(), White()); return;}
	
	Session& ses = *this->ses;
	SessionData& d = ses.Data();
	
	Size sz = GetSize();
	int vis_len = min(sz.cx, sz.cy);
	int density = 5;
	
	int count = vis_len / density;
	if (count < 2) {
		draw.DrawRect(sz, White());
		return;
	}
	
	int x_off = (sz.cx - vis_len) / 2;
	int y_off = (sz.cy - vis_len) / 2;
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	
	ses.Enter();
	Net& net = ses.GetNetwork();
	
	LayerBase* input = ses.GetInput();
	if (!input) {
		ses.Leave();
		return;
	}
	int data_count = d.GetDataCount();
	double offset = max(max(-d.GetMin(0), -d.GetMin(1)), max(d.GetMax(0), d.GetMax(1)));
	offset *= 1.1;
	double diff = offset * 2;
	double step = diff / (count - 1);
	
	this->offset = offset;
	this->vis_len = vis_len;
	
	
	Volume netx(1,1,2,0);
	Color clr_a(250, 150, 150);
	Color clr_b(150, 250, 150);
	
	double y = -offset;
	int scr_y = 0;
	
	int k = 0;
	for (int i = 0; i < count; i++) {
		
		double x = -offset;
		int scr_x = 0;
		
		for (int j = 0; j < count; j++) {
			
			netx.Set(0,0,0, x);
			netx.Set(0,0,1, y);
			
			Volume& a = net.Forward(netx, false);
			
			double aw0 = a.Get(0,0,0);
			double aw1 = a.Get(0,0,1);
			
			int label = aw0 > aw1;
			
			id.DrawRect(x_off + scr_x, y_off + scr_y, density, density, label ? clr_a : clr_b);
			
			scr_x += density;
			x += step;
			k++;
		}
		
		scr_y += density;
		y += step;
	}
	
	Color clr_a2(100,200,100);
	Color clr_b2(200,100,100);
	double radius = 13.0;
	double radius_2 = radius / 2.0;
	for(int i = 0; i < data_count; i++) {
		double x = d.GetData(i, 0);
		double y = d.GetData(i, 1);
		int label = d.GetLabel(i);
		double x_fac = (x + offset) / diff;
		double y_fac = (y + offset) / diff;
		int xi = max(0, min(count-1, (int)(x_fac * count)));
		int yi = max(0, min(count-1, (int)(y_fac * count)));
		int pos = yi * count + xi;
		int scr_x = x_fac * vis_len;
		int scr_y = y_fac * vis_len;
		id.DrawEllipse(x_off + scr_x - radius_2, y_off + scr_y - radius_2, radius, radius, label ? clr_a2 : clr_b2, 1, Black());
	}
	
	ses.Leave();
	
	draw.DrawImage(0, 0, id);
}

void PointCtrl::LeftDown(Point p, dword keyflags) {
	
	
}

}
