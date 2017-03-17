#include "PuckWorld.h"

PuckWorldCtrl::PuckWorldCtrl() {
	agent = NULL;
	
}

void PuckWorldCtrl::Layout() {
	
}

void PuckWorldCtrl::MouseWheel(Point, int zdelta, dword) {
	
}

void PuckWorldCtrl::LeftDown(Point p, dword keyflags) {
	
}

void PuckWorldCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	int W = sz.cx;
	int H = sz.cy;
	
	if (!agent) {w.DrawRect(sz, White()); return;}
	
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	
	
	
	Color fill, stroke;
	double rad, rad2;
	
	// reflect puck world state on screen
	double ppx	= agent->ppx;
	double ppy	= agent->ppy;
	double tx	= agent->tx;
	double ty	= agent->ty;
	double tx2	= agent->tx2;
	double ty2	= agent->ty2;
	
	
	// bad target
	stroke = Color(0, 0, 0);
	fill = Color(255, 229, 229);
	rad = agent->BADRAD * H;
	rad2 = rad * 2;
	id.DrawEllipse(tx2*W - rad, ty2*H - rad, rad2, rad2, fill, 1, stroke);
	
	fill = Color(255, 0, 0);
	rad = 10;
	rad2 = rad * 2;
	id.DrawEllipse(tx2*W - rad, ty2*H - rad, rad2, rad2, fill, 1, stroke);
	
	
	// draw the target
	rad = 10;
	rad2 = rad * 2;
	fill = Color(0, 255, 0);
	id.DrawEllipse(tx*W - rad, ty*H - rad, rad2, rad2, fill, 1, stroke);
	
	// color agent by reward
	double vv = agent->reward + 0.5;
	double ms = 255.0;
	int diff = min(255, (int)(vv*ms));
	int r, g, b;
	if (vv > 0)	{ g = 255; r = 255 - diff; b = 255 - diff; }
	else		{ g = 255 + diff; r = 255; b = 255 + diff; }
	fill = Color(r, g, b);
	
	
	// draw the puck
	int x = ppx*W;
	int y = ppy*H;
	rad = agent->rad * W;
	rad2 = rad * 2;
	id.DrawEllipse(x - rad, y - rad, rad2, rad2, fill, 1, stroke);
	
	
	int action = agent->action;
	int x2 = x;
	int y2 = y;
	int af = 20;
	if (action == ACT_LEFT) {
		x2 = x - af;
	}
	else if (action == ACT_RIGHT) {
		x2 = x + af;
	}
	else if (action == ACT_UP) {
		y2 = y - af;
	}
	else if (action == ACT_DOWN) {
		y2 = y + af;
	}
	
	id.DrawLine(x, y, x2, y2, 1, Black());
	
	
	
	w.DrawImage(0, 0, id);
}

