#include "ConvNetCtrl.h"

namespace ConvNetCtrl {

LayerView::LayerView(LayerCtrl* lc) : lc(lc) {
	
}
	
void LayerView::Paint(Draw& d) {
	
	Size sz = GetSize();
	d.DrawRect(sz, White());
	
	Session& ses = *lc->ses;
	int d0 = lc->d0;
	int d1 = lc->d1;
	bool sync = lc->sync_trainer;
	
	int vis_len = min(sz.cx, sz.cy);
	int density = 20;
	int density_2 = density / 2;
	int x_off = (sz.cx - vis_len) / 2;
	int y_off = (sz.cy - vis_len) / 2;
	
	int count = vis_len / density;
	if (count < 2) return;
	
	if (sync) ses.Enter();
	
	Net& net = ses.GetNetwork();
	
	
	InputLayer* input = ses.GetInput();
	if (!input) {
		ses.Leave();
		return;
	}
	
	int count2 = count * count;
	gridx.SetCount(count2);
	gridy.SetCount(count2);
	gridl.SetCount(count2);
	
	int line_count = count * 2;
	if (lines.GetCount() != line_count) {
		lines.SetCount(line_count);
		for(int i = 0; i < lines.GetCount(); i++)
			lines[i].SetCount(count);
	}
	
	int data_count = ses.GetDataCount();
	double offset = max(max(-ses.GetMin(0), -ses.GetMin(1)), max(ses.GetMax(0), ses.GetMax(1)));
	offset *= 1.1;
	double diff = offset * 2;
	double step = diff / (count - 1);
	
	LayerBase& lb = *net.GetLayers()[lc->lix];
	Volume& output = lb.output_activation;
	
	Volume netx(1,1,2,0);
	Color clr_a(250, 150, 150);
	Color clr_b(150, 250, 150);
	
	double min_x = +DBL_MAX;
	double min_y = +DBL_MAX;
	double max_x = -DBL_MAX;
	double max_y = -DBL_MAX;
	
	int k = 0;
	double y = -offset;
	
	for(int i = 0; i < count; i++) {
		
		double x = -offset;
		
		for(int j = 0; j < count; j++) {
			
			netx.Set(0,0,0, x);
			netx.Set(0,0,1, y);
			
			Volume& a = net.Forward(netx, false); // modifies 'output'
			
			double aw0 = a.Get(0,0,0);
			double aw1 = a.Get(0,0,1);
			gridl[k] = aw0 > aw1;
			
			double xt = output.Get(0,0,d0); // in screen coords
			double yt = output.Get(0,0,d1); // in screen coords
			
			max_x = max(max_x, xt);
			max_y = max(max_y, yt);
			min_x = min(min_x, xt);
			min_y = min(min_y, yt);
			
			gridx[k] = xt;
			gridy[k] = yt;
			
			k++;
			x += step;
		}
		
		y += step;
	}
	
	double diff_x = max_x - min_x;
	double diff_y = max_y - min_y;
	
	k = 0;
	int l = 0;
	for(int i = 0; i < count; i++) {
		
		Vector<Point>& right_line = lines[i];
		
		for(int j = 0; j < count; j++) {
			
			Vector<Point>& down_line = lines[count + j];
			
			double x0 = (gridx[k] - min_x) / diff_x * vis_len;
			double y0 = (gridy[k] - min_y) / diff_y * vis_len;
			int label = gridl[k];
			
			Point& ptd = down_line[i];
			ptd.x = x_off + x0;
			ptd.y = y_off + y0;
			
			Point& ptr = right_line[j];
			ptr.x = x_off + x0;
			ptr.y = y_off + y0;
			
			d.DrawRect(x_off + x0 - density_2, y_off + y0 - density_2, density, density, label ? clr_a : clr_b);
			
			k++;
		}
	}
	
	Color stroke_clr = GrayColor(0);
	
	for(int i = 0; i < lines.GetCount(); i++) {
		const Vector<Point>& l = lines[i];
		d.DrawPolyline(l, 1, stroke_clr);
	}
	
	
	Color clr_a2(100,200,100);
	Color clr_b2(200,100,100);
	double radius = 13.0;
	double radius_2 = radius / 2.0;
	
	for(int i = 0; i < data_count; i++) {
		
		// also draw transformed data points while we're at it
		netx.Set(0, 0, 0, ses.GetData(i, 0));
		netx.Set(0, 0, 1, ses.GetData(i, 1));
		Volume& a = net.Forward(netx, false);
		
		double scr_x = (output.Get(0,0,d0) - min_x) / diff_x * vis_len;
		double scr_y = (output.Get(0,0,d1) - min_y) / diff_y * vis_len;
		int label = ses.GetLabel(i);
		
		d.DrawEllipse(x_off + scr_x - radius_2, y_off + scr_y - radius_2, radius, radius, label ? clr_a2 : clr_b2);
	}
	
	if (sync) ses.Leave();
}







LayerCtrl::LayerCtrl(Session& ses) : view(this) {
	this->ses = &ses;
	d0 = 0;
	d1 = 1;
	lix = 1;
	sync_trainer = true;
	
	
	Add(view.VSizePos(0,60).HSizePos());
	Add(btn_cycle.BottomPos(30,30).LeftPos(0,150));
	Add(lbl_layer.BottomPos(30,30).HSizePos(155));
	Add(layerbtn_split.BottomPos(0,30).HSizePos());
	
	btn_cycle.SetLabel("Cycle neurons");
	btn_cycle <<= THISBACK(Cycle);
	lbl_layer.SetLabel("drawing neurons 0 and 1 of layer with index 0");
	
	ses.WhenSessionLoaded << THISBACK(PostRefreshData);
	
}

void LayerCtrl::RefreshData() {
	layerbtn_split.Clear();
	layer_buttons.Clear();
	
	ses->Enter();
	Net& net = ses->GetNetwork();
	
	const Vector<LayerBasePtr>& layers = net.GetLayers();
	if (lix < 0 || lix >= layers.GetCount()) lix = 0;
	d0 = 0;
	d1 = 1;
	
	for(int i = 0; i < layers.GetCount(); i++) {
		const LayerBase& lb = *layers[i];
		
		String lstr = lb.GetKey();
		
		Button& b = layer_buttons.Add();
		b.SetLabel(lstr);
		b <<= THISBACK1(ViewLayer, i);
		layerbtn_split << b;
	}
	ses->Leave();
	
	view.Refresh();
	RefreshCycle();
}

void LayerCtrl::ViewLayer(int i) {
	lix = i;
	d0 = 0; // first dimension to show visualized
	d1 = 1; // second dimension to show visualized
	view.Refresh();
	RefreshCycle();
}

void LayerCtrl::Cycle() {
	Net& net = ses->GetNetwork();
	const Vector<LayerBasePtr>& layers = net.GetLayers();
	LayerBase& lb = *layers[lix];
	
	d0 += 1;
	d1 += 1;
	if(d1 >= lb.output_depth) d1 = 0; // and wrap
	if(d0 >= lb.output_depth) d0 = 0; // and wrap
	view.Refresh();
	
	RefreshCycle();
}

void LayerCtrl::RefreshCycle() {
	Net& net = ses->GetNetwork();
	String s;
	s << "drawing neurons " << d0 << " and " << d1 << " of layer #" << lix << " (" << net.GetLayers()[lix]->GetKey() << ")";
	lbl_layer.SetLabel(s);
}


}
