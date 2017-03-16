#include "ConvNetCtrl.h"

namespace ConvNet {

GridWorldCtrl::GridWorldCtrl() {
	agent = NULL;
	selected = -1;
}

void GridWorldCtrl::Layout() {
	
}

void GridWorldCtrl::MouseWheel(Point, int zdelta, dword) {
	
}

void GridWorldCtrl::LeftDown(Point p, dword keyflags) {
	
	if (agent) {
		Size sz = GetSize();
		int gh = agent->height; // height in cells
		int gw = agent->width; // width in cells
		int max_size = min(sz.cx, sz.cy);
		int xoff = sz.cx / 2 - max_size / 2;
		int yoff = sz.cy / 2 - max_size / 2;
		double cs = (double)max_size / max(gh, gw);
		
		int x = (p.x - xoff) / cs;
		int y = (p.y - yoff) / cs;
		
		if (x >= 0 && x < gw && y >= 0 && y < gh) {
			selected = agent->GetPos(x, y, 0);
			Refresh();
			WhenGridFocus();
		}
		else {
			selected = -1;
			Refresh();
			WhenGridUnfocus();
		}
	}
}

void GridWorldCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	
	if (!agent) {w.DrawRect(sz, White()); return;}
	
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	
	int gh = agent->height; // height in cells
	int gw = agent->width; // width in cells
	
	int max_size = min(sz.cx, sz.cy);
	int xoff = sz.cx / 2 - max_size / 2;
	int yoff = sz.cy / 2 - max_size / 2;
	double cs = (double)max_size / max(gh, gw);
	
	int small_fnt_h = max(7.0, cs / 6.0);
	Font small_fnt = Arial(small_fnt_h);
	
	int med_fnt_h = max(7.0, cs / 4.0);
	Font med_fnt = Arial(med_fnt_h);
	
	int tip = max(3.0, cs / 24.0);
	int tip2 = tip * 2;
	
	Color line_clr = GrayColor(128);
	
	int x1 = xoff + gw * cs;
	int y1 = yoff + gh * cs;
	
	// updates the grid with current state of world/agent
	Vector<Point> tipvec;
	Color arrow = Black();
	for (int y = 0; y < gh; y++) {
		for (int x = 0; x < gw; x++) {
			int x0 = xoff + x * cs;
			int y0 = yoff + y * cs;
			int r = 255, g = 255, b = 255;
			int s = agent->GetPos(x, y, 0);
			
			double vv = agent->value[s];
			int ms = 100;
			if (vv > 0)		{ g = 255; r = 255 - vv*ms; b = 255 - vv*ms; }
			if (vv < 0)		{ g = 255 + vv*ms; r = 255; b = 255 + vv*ms; }
			
			Color vcol(r, g, b);
			Color rcol = vcol;
			if (agent->disable[s] == 1)	{ vcol = GrayColor(0xAA); rcol = GrayColor(0xAA); }
			
			// update colors of rectangles based on value
			Color fill_clr = s == selected ? Color(255,255,0) : vcol;
			id.DrawRect(x0, y0, cs, cs, fill_clr);
			
			// skip rest for cliff
			if (agent->disable[s] == 1) continue;
			
			// write reward texts
			double rv = agent->reward[s];
			if (rv != 0) {
				String rstr = "R " + FormatDoubleFix(rv, 1, FD_ZEROS);
				id.DrawText(x0 + 4, y0 + cs - small_fnt_h - 2, rstr, small_fnt);
			}
			
			// write value
			double tv = agent->value[s];
			id.DrawText(x0 + 4, y0 + 4, FormatDoubleFix(tv, 2, FD_ZEROS), med_fnt);
			
			// update policy arrows
			for (int a = 0; a < 4; a++) {
				double prob = agent->poldist[a][s];
				if (prob <= 0.0) continue;
				
				double ss = cs/2 * prob * 0.9;
				int nx, ny, x1,y1, xtrim = 1, ytrim = 1;
				tipvec.SetCount(0);
				if (a == ACT_LEFT)	{
					xtrim = 0;
					nx=-ss;				ny=0;		x1 = x0+cs/2+nx;	y1 = y0+cs/2+ny;
					tipvec << Point(x1, y1) << Point(x1+tip2, y1-tip) << Point(x1+tip2, y1+tip);
				}
				else if (a == ACT_DOWN)	{
					nx=0;				ny=-ss;		x1 = x0+cs/2+nx;	y1 = y0+cs/2+ny;
					tipvec << Point(x1, y1) << Point(x1+tip, y1+tip2) << Point(x1-tip, y1+tip2);
				}
				else if (a == ACT_UP)	{
					ytrim = 0;
					nx=0;				ny=ss;		x1 = x0+cs/2+nx;	y1 = y0+cs/2+ny;
					tipvec << Point(x1, y1) << Point(x1+tip, y1-tip2) << Point(x1-tip, y1-tip2);
				}
				else if (a == ACT_RIGHT)	{
					nx=ss;				ny=0;		x1 = x0+cs/2+nx;	y1 = y0+cs/2+ny;
					tipvec << Point(x1, y1) << Point(x1-tip2, y1-tip) << Point(x1-tip2, y1+tip);
				}
				else continue;
				
				#ifdef flagWIN32
				// This is small but annoying difference in windows
				xtrim = 0;
				ytrim = 0;
				#endif
				
				id.DrawLine(
					x0+cs/2-xtrim,
					y0+cs/2-ytrim,
					x1-xtrim,
					y1-ytrim,
					2, arrow);
				
				id.DrawPolygon(tipvec, arrow, 0, arrow);
				
			}
		}
	}
	
	
	for (int y = 0; y <= gh; y++) {
		for (int x = 0; x <= gw; x++) {
			int x0 = xoff + x * cs;
			int y0 = yoff + y * cs;
			id.DrawLine(x0-1, yoff, x0-1, y1, 3, line_clr);
			id.DrawLine(xoff, y0-1, x1, y0-1, 3, line_clr);
		}
	}
	
	w.DrawImage(0, 0, id);
	
}


}
