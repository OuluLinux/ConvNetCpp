#include "ConvNetCtrl.h"

namespace ConvNet {

LayerView::LayerView(LayerCtrl* lc) : lc(lc) {
	
}

void LayerView::ClearCache() {
	labels.Clear();
	volumes.Clear();
	tmp_imgs.Clear();
	lbl_colors.Clear();
}
	
void LayerView::Paint(Draw& d) {
	if (!lc->ses) {
		d.DrawRect(GetSize(), White());
		return;
	}
	
	Session& ses = *lc->ses;
	lix = lc->lix;
	d0 = lc->d0;
	d1 = lc->d1;
	
	Size sz = GetSize();
	vis_len = min(sz.cx, sz.cy);
	density = 20;
	count = vis_len / density;
	if (count < 2) {
		d.DrawRect(sz, White());
		return;
	}
	
	density_2 = density / 2;
	x_off = (sz.cx - vis_len) / 2;
	y_off = (sz.cy - vis_len) / 2;
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	
	ses.Enter();
	Net& net = ses.GetNetwork();
	
	
	InputLayer* input = ses.GetInput();
	if (!input) {
		d.DrawRect(sz, White());
		ses.Leave();
		return;
	}
	
	// Select drawing by input length
	Volume& in = input->output_activation;
	
	
	// 1D x input... 
	if (in.GetLength() == 1)
		PaintInputX(id);
	
	// 2D x,y input...
	else if (in.GetLength() == 2)
		PaintInputXY(id);
	
	// Image input, grayscale or color
	else if (in.GetWidth() > 1 && in.GetHeight() > 1)
		PaintInputImage(id);
	
	
	ses.Leave();
	
	d.DrawImage(0, 0, id);
}

void LayerView::PaintInputX(Draw& id) {
	Session& ses = *lc->ses;
	SessionData& d = ses.Data();
	Net& net = ses.GetNetwork();
	LayerBase& layer = *net.GetLayers()[lix];
	
	Size sz = GetSize();
	Volume netx(1,1,1,0);
	
	double min_x = +DBL_MAX;
	double max_x = -DBL_MAX;
	double min_y = +DBL_MAX;
	double max_y = -DBL_MAX;
	
	#define X(v) (v - min_x) / diff_x * sz.cx
	#define Y(v) (v - min_y) / diff_y * sz.cy
	
	int data_count = d.GetDataCount();
	for (int i = 0; i < data_count; i++) {
		double x = d.GetData(i, 0);
		double y = d.GetResult(i).Get(0);
		min_x = min(min_x, x);
		min_y = min(min_y, y);
		max_x = max(max_x, x);
		max_y = max(max_y, y);
	}
	double diff_x = max_x - min_x;
	double diff_y = max_y - min_y;

	// draw axes
	id.DrawLine(0, Y(0) - 1, sz.cx, Y(0) - 1, 3, GrayColor());
	id.DrawLine(X(0) - 1, 0, X(0) - 1, sz.cy, 3, GrayColor());
	
	// draw decisions in the grid
	double density= 5.0;
	bool draw_neuron_outputs = true;
	
	// draw final decision
	Vector<VolumeDataBase*> neurons;
	tmp_pts1.SetCount(0);
	
	int neuron_count = layer.output_activation.GetLength();
	tmp_pts2.SetCount(neuron_count);
	for(int i = 0; i < tmp_pts2.GetCount(); i++)
		tmp_pts2[i].SetCount(0);
	
	for (double x = 0.0; x <= sz.cx; x += density) {
		
		netx.Set(0, x / sz.cx * diff_x + min_x);
		
		Volume& a = net.Forward(netx);
		double y = Y(a.Get(0));
		
		// draw individual neurons on first layer
		if (draw_neuron_outputs) {
			Volume& out = layer.output_activation;
			for (int i = 0; i < out.GetLength(); i++)
				tmp_pts2[i].Add(Pointf(x, Y(out.Get(i))));
		}
		
		tmp_pts1.Add(Pointf(x, y));
	}
	
	for(int i = 0; i < tmp_pts2.GetCount(); i++)
		id.DrawPolyline(tmp_pts2[i], 1, Color(250,50,50));
	
	id.DrawPolyline(tmp_pts1, 1, Black());
	
	// draw datapoints. Draw support vectors larger
	int radius = 10;
	int radius_2 = radius / 2;
	for (int i = 0; i < data_count; i++) {
		double x = X(d.GetData(i, 0))			- radius_2;
		double y = Y(d.GetResult(i).Get(0))	- radius_2;
		id.DrawEllipse(x, y, radius, radius, Black());
	}
	
	Font font = Arial(16).Bold();
	id.DrawText(5, 5, "Average loss: " + DblStr(ses.GetLossAverage()), font, Blue());
	
}

void LayerView::PaintInputXY(Draw& id) {
	Session& ses = *lc->ses;
	SessionData& d = ses.Data();
	Net& net = ses.GetNetwork();
	
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
	
	int data_count = d.GetDataCount();
	double offset = max(max(-d.GetMin(0), -d.GetMin(1)), max(d.GetMax(0), d.GetMax(1)));
	offset *= 1.1;
	double diff = offset * 2;
	double step = diff / (count - 1);
	
	const Vector<LayerBasePtr>& layers = net.GetLayers();
	if (lix < 0 || lix >= layers.GetCount()) {return;}
	LayerBase& lb = *layers[lix];
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
			
			id.DrawRect(x_off + x0 - density_2, y_off + y0 - density_2, density, density, label ? clr_a : clr_b);
			
			k++;
		}
	}
	
	Color stroke_clr = GrayColor(0);
	
	for(int i = 0; i < lines.GetCount(); i++) {
		const Vector<Point>& l = lines[i];
		id.DrawPolyline(l, 1, stroke_clr);
	}
	
	
	Color clr_a2(100,200,100);
	Color clr_b2(200,100,100);
	double radius = 13.0;
	double radius_2 = radius / 2.0;
	
	for(int i = 0; i < data_count; i++) {
		
		// also draw transformed data points while we're at it
		netx.Set(0, 0, 0, d.GetData(i, 0));
		netx.Set(0, 0, 1, d.GetData(i, 1));
		Volume& a = net.Forward(netx, false);
		
		double scr_x = (output.Get(0,0,d0) - min_x) / diff_x * vis_len;
		double scr_y = (output.Get(0,0,d1) - min_y) / diff_y * vis_len;
		int label = d.GetLabel(i);
		
		id.DrawEllipse(x_off + scr_x - radius_2, y_off + scr_y - radius_2, radius, radius, label ? clr_a2 : clr_b2, 1, Black());
	}
	
}

void LayerView::PaintInputImage(Draw& id) {
	Session& ses = *lc->ses;
	SessionData& d = ses.Data();
	Net& net = ses.GetNetwork();
	
	int data_w = d.GetDataWidth();
	int data_h = d.GetDataHeight();
	int data_d = d.GetDataDepth();
	bool is_color = data_d == 3;
	
	if (volumes.IsEmpty()) {
		d.GetUniformClassData(16, volumes, labels);
		tmp_imgs.SetCount(volumes.GetCount());
		for(int i = 0; i < volumes.GetCount(); i++) {
			VolumeDataBase& data = *volumes[i];
			Image& img = tmp_imgs[i];
			ImageBuffer ib(data_w, data_h);
			RGBA* it = ib.Begin();
			if (is_color) {
				for (int y = 0; y < data_h; y++) {
					for (int x = 0; x < data_w; x++) {
						it->r = data.Get(x, y, 0, data_w, data_d) * 255;
						it->g = data.Get(x, y, 1, data_w, data_d) * 255;
						it->b = data.Get(x, y, 2, data_w, data_d) * 255;
						it->a = 255;
						it++;
					}
				}
			} else {
				for (int y = 0; y < data_h; y++) {
					for (int x = 0; x < data_w; x++) {
						it->r = data.Get(x, y, 0, data_w, data_d) * 255;
						it->g = it->r;
						it->b = it->r;
						it->a = 255;
						it++;
					}
				}
			}
			img = ib;
		}
		
		
		int cls_count = d.GetClassCount();
		lbl_colors.SetCount(cls_count);
		for(int i = 0; i < cls_count; i++) {
			lbl_colors[i] = Rainbow((double)i / cls_count);
		}
	}
	
	const Vector<LayerBasePtr>& layers = net.GetLayers();
	if (lix < 0 || lix >= layers.GetCount()) return;
	LayerBase& lb = *layers[lix];
	Volume& output = lb.output_activation;
	
	
	Volume netx(data_w, data_h, data_d, 0);
	
	tmp_pts.SetCount(volumes.GetCount());
	
	double min_x = +DBL_MAX;
	double min_y = +DBL_MAX;
	double max_x = -DBL_MAX;
	double max_y = -DBL_MAX;
	
	for(int i = 0; i < volumes.GetCount(); i++) {
		
		// also draw transformed data points while we're at it
		netx.SetData(*volumes[i]);
		Volume& a = net.Forward(netx, false);
		
		Pointf& p = tmp_pts[i];
		p.x = output.Get(0,0,d0);
		p.y = output.Get(0,0,d1);
		if (p.x < min_x) min_x = p.x;
		if (p.y < min_y) min_y = p.y;
		if (p.x > max_x) max_x = p.x;
		if (p.y > max_y) max_y = p.y;
	}
	
	double diff_x = max_x - min_x;
	double diff_y = max_y - min_y;
	
	int w_2 = data_w / 2;
	int h_2 = data_h / 2;
	
	for(int i = 0; i < tmp_pts.GetCount(); i++) {
		Pointf& p = tmp_pts[i];
		
		double scr_x = (p.x - min_x) / diff_x * vis_len;
		double scr_y = (p.y - min_y) / diff_y * vis_len;
		int label = labels[i];
		
		id.DrawRect(
			x_off + scr_x - w_2 - 2,
			y_off + scr_y - h_2 - 2,
			data_w + 4,
			data_h + 4,
			lbl_colors[label]);
		id.DrawImage(
			x_off + scr_x - w_2,
			y_off + scr_y - h_2,
			tmp_imgs[i]);
	}
	
}













LayerCtrl::LayerCtrl() : view(this) {
	ses = NULL;
	d0 = 0;
	d1 = 1;
	lix = 1;
	
	
	Add(view.VSizePos(0,60).HSizePos());
	Add(btn_cycle.BottomPos(30,30).LeftPos(0,150));
	Add(lbl_layer.BottomPos(30,30).HSizePos(155));
	Add(layerbtn_split.BottomPos(0,30).HSizePos());
	
	btn_cycle.SetLabel("Cycle neurons");
	btn_cycle <<= THISBACK(Cycle);
	lbl_layer.SetLabel("drawing neurons 0 and 1 of layer with index 0");
	
}

void LayerCtrl::SetSession(Session& ses) {
	this->ses = &ses;
	ses.WhenSessionLoaded << THISBACK(PostRefreshData);
}

void LayerCtrl::RefreshData() {
	if (!ses) return;
	
	layerbtn_split.Clear();
	layer_buttons.Clear();
	
	ses->Enter();
	Net& net = ses->GetNetwork();
	
	const Vector<LayerBasePtr>& layers = net.GetLayers();
	if (lix < 0 || lix >= layers.GetCount()) lix = 0;
	d0 = 0;
	d1 = 1;
	
	int first_xy_layer = -1;
	for(int i = 0; i < layers.GetCount(); i++) {
		const LayerBase& lb = *layers[i];
		
		String lstr = lb.GetKey();
		int length = lb.output_depth * lb.output_height * lb.output_width;
		lstr << "(" << length << ")";
		
		if (i && first_xy_layer == -1 && length == 2)
			first_xy_layer = i;
		
		ButtonOption& b = layer_buttons.Add();
		b.SetLabel(lstr);
		b <<= THISBACK1(ViewLayer, i);
		layerbtn_split << b;
	}
	ses->Leave();
	
	view.ClearCache();
	view.Refresh();
	RefreshCycle();
	
	PostCallback(THISBACK1(ViewLayer, first_xy_layer != -1 ? first_xy_layer : 1));
}

void LayerCtrl::ViewLayer(int i) {
	if (layer_buttons.IsEmpty()) return;
	i = max(0, min(layer_buttons.GetCount()-1, i));
	
	for(int i = 0; i < layer_buttons.GetCount(); i++)
		layer_buttons[i].Set(false);
	lix = i;
	d0 = 0; // first dimension to show visualized
	d1 = 1; // second dimension to show visualized
	layer_buttons[i].Set(true);
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
