#include "ConvNetCtrl.h"

namespace ConvNet {

ConvLayerCtrl::ConvLayerCtrl() {
	ses = NULL;
	layer_id = -1;
	height = 0;
	is_color = false;
	hide_gradients = false;
}

void ConvLayerCtrl::Paint(Draw& d) {
	Size sz = GetSize();
	
	if (!ses) {d.DrawRect(sz, White()); return;}
	
	//ses->Enter();
	Net& net = ses->GetNetwork();
	if (layer_id < 0 || layer_id >= net.GetLayers().GetCount()) {
		ses->Leave();
		d.DrawRect(sz, White());
		return;
	}
	
	ImageDraw id(sz);
	
	PaintSize(id, sz);
	
	//ses->Leave();
	
	d.DrawImage(0, 0, id);
}

void ConvLayerCtrl::PaintSize(Draw& id, Size sz) {
	
	// print some stats on left of the layer
	Net& net = ses->GetNetwork();
	LayerBase& l = *net.GetLayers()[layer_id];
	int xoff = sz.cx * 0.4;
	int y = 0;
	String type = l.GetKey();
	int hash = type.GetHashValue();
	int r = 128 + 64 + 32 + ((hash & 0xFF0000) >> 16) / 8;
	int g = 128 + 64 + 32 + ((hash & 0xFF00) >> 8) / 8;
	int b = 128 + 64 + 32 + ((hash & 0xFF) >> 0) / 8;
	Color bg(r, g, b);
	id.DrawRect(sz, bg);
	
	
	String title;
	title << type << "(" << l.output_width << "," << l.output_height << "," << l.output_depth << ")";
	Font big_fnt = ArialZ(15);
	Font fnt = ArialZ(11);
	Size txt_sz = GetTextSize(title, big_fnt);
	id.DrawText(2, 0, title, big_fnt);
	y += txt_sz.cy;
	
	
	if (type == "conv") {
		ConvLayer& conv = dynamic_cast<ConvLayer&>(l);
		String s = "filter size ";
		if (!conv.filters.IsEmpty()) {
			Volume& f0 = conv.filters[0];
			s << f0.GetWidth() << "x" <<
				 f0.GetHeight() << "x" <<
				 f0.GetDepth() << ", stride " << conv.GetStride();
		}
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(2, y, s, fnt);
		y += txt_sz.cy;
		
		int tot_params = conv.filters.GetCount() * conv.width * conv.height * conv.input_depth + conv.filters.GetCount();
		s.Clear();
		s << "parameters: "
			<< conv.filters.GetCount() << " * " << conv.width << " * " << conv.height << " * " << conv.input_depth << " + " << conv.filters.GetCount()
			<< " = " << tot_params;
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(2, y, s, fnt);
		y += txt_sz.cy;
		
	}
	else if (type == "pool") {
		PoolLayer& pool = dynamic_cast<PoolLayer&>(l);
		String s;
		s << "pooling size " <<
			pool.width << "x" << pool.height << ", stride " << pool.stride;
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(2, y, s, fnt);
		y += txt_sz.cy;
	}
	else if (type == "fc") {
		FullyConnLayer& fc = dynamic_cast<FullyConnLayer&>(l);
		
		int tot_params = fc.filters.GetCount() * fc.GetInputCount() + fc.filters.GetCount();
		String s;
		s << "parameters: "
			<< fc.filters.GetCount() << " * " << fc.GetInputCount() << " + " << fc.filters.GetCount()
			<< " = " << tot_params;
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(2, y, s, fnt);
		y += txt_sz.cy;
	}
	
	// find min, max activations and display them
	{
		Volume& v = l.output_activation;
		double min = +DBL_MAX;
		double max = -DBL_MAX;
		for(int i = 0; i < v.GetLength(); i++) {
			double d = v.Get(i);
			if (d > max) max = d;
			if (d < min) min = d;
		}

		String s = Format("max activation: %8,n, min: %8,n", max, min);
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(2, y, s, fnt);
		y += txt_sz.cy;
	}
	
	if (!hide_gradients) {
		Volume& v = l.output_activation;
		double min = +DBL_MAX;
		double max = -DBL_MAX;
		for(int i = 0; i < v.GetLength(); i++) {
			double d = v.GetGradient(i);
			if (d > max) max = d;
			if (d < min) min = d;
		}

		String s = Format("max gradient: %8,n, min: %8,n", max, min);
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(2, y, s, fnt);
		y += txt_sz.cy;
	}
	
	int left_height = y;
	
	
	// visualize activations
	Point pt(xoff, 4);
	if (type == "regression") {
		int scale = 2;
		String s;
		s << "Activations:";
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(pt.x, pt.y, s, fnt);
		pt.y += txt_sz.cy;
		int depth = l.output_activation.GetDepth();
		int w = sqrt((double)depth);
		int h = w ? depth / w : 0;
		int w_clr = sqrt((double)depth / 3);
		int h_clr = w_clr ? depth / 3 / w_clr : 0;
		if (w * h == depth && l.output_activation.GetWidth() == 1 && l.output_activation.GetHeight() == 1) {
			Volume v(w, h, 1, l.output_activation);
			DrawActivations(id, sz, pt, v, scale, false);
		}
		else if (w_clr * h_clr * 3 == depth && l.output_activation.GetWidth() == 1 && l.output_activation.GetHeight() == 1) {
			Volume v(w_clr, h_clr, 3, l.output_activation);
			DrawActivations(id, sz, pt, v, scale, false);
		}
		else {
			DrawActivations(id, sz, pt, l.output_activation, scale, false);
		}
	}
	else {
		int scale = 2;
		if (type == "softmax" || type == "fc")
			scale = 10; // for softmax
		String s;
		s << "Activations:";
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(pt.x, pt.y, s, fnt);
		pt.y += txt_sz.cy;
		DrawActivations(id, sz, pt, l.output_activation, scale, false);
	}
	
	
	// visualize data gradients
	if (!hide_gradients && type != "softmax" && type != "regression") {
		int scale = 2;
		if (type == "softmax" || type == "fc")
			scale = 10; // for softmax
		String s;
		s << "Activation Gradients:";
		txt_sz = GetTextSize(s, fnt);
		id.DrawText(pt.x, pt.y, s, fnt);
		pt.y += txt_sz.cy;
		DrawActivations(id, sz, pt, l.output_activation, scale, true);
	}
	
	
	// visualize filters if they are of reasonable size
	if (type == "conv") {
		ConvLayer& conv = dynamic_cast<ConvLayer&>(l);
		
		if (!conv.filters.IsEmpty() && conv.filters[0].GetWidth() > 3) {
			int count = conv.filters.GetCount();
			
			// actual weights
			String s;
			s << "Weights:";
			txt_sz = GetTextSize(s, fnt);
			id.DrawText(pt.x, pt.y, s, fnt);
			pt.x = xoff;
			pt.y += txt_sz.cy;
			for (int j = 0; j < count; j++) {
				DrawActivations(id, sz, pt, conv.filters[j], 2, false, j == count-1);
			}
			pt.x = xoff;
			
			// gradients
			s.Clear();
			s << "Weight Gradients:";
			txt_sz = GetTextSize(s, fnt);
			id.DrawText(pt.x, pt.y, s, fnt);
			pt.y += txt_sz.cy;
			for (int j = 0; j < count; j++) {
				DrawActivations(id, sz, pt, conv.filters[j], 2, true, j == count-1);
			}
		}
	}
	
	
	// Set require height for ctrl
	height = max(left_height, pt.y) + 3;
}

// elt is the element to add all the canvas activation drawings into
// A is the Vol() to use
// scale is a multiplier to make the visualizations larger. Make higher for larger pictures
// if grads is true then gradients are used instead
void ConvLayerCtrl::DrawActivations(Draw& draw, Size& sz, Point& pt, Volume& v, int scale, bool draw_grads, bool end_newline) {
	int x = sz.cx * 0.4;
	
	int s = scale > 0 ? scale : 2; // scale
	
	// get max and min activation to scale the maps automatically
	double min = +DBL_MAX;
	double max = -DBL_MAX;
	for(int i = 0; i < v.GetLength(); i++) {
		double d = draw_grads ? v.GetGradient(i) : v.Get(i);
		if (d > max) max = d;
		if (d < min) min = d;
	}
	double diff = max - min;
	
	int W = v.GetWidth() * s;
	int H = v.GetHeight() * s;
	
	
	// create the canvas elements, draw and add to DOM
	bool is_color = this->is_color;
	
	// Looking "colored" data is a lucky guess, but the visual part is almost irrelevant anyway
	if (v.GetDepth() < 3 || v.GetDepth() % 3 != 0) is_color = false;
	
	for (int d = 0; d < v.GetDepth(); d += is_color ? 3 : 1) {
		ImageBuffer ib(W, H);
		RGBA* it = ib.Begin();
		bool has_white = false;
		
		for(int y = 0; y < v.GetHeight(); y++) {
			for (int x = 0; x < v.GetWidth(); x++) {
				
				// Grayscale image
				if (!is_color) {
					byte dval;
					if (draw_grads) {
						dval = (v.GetGradient(x, y, d) - min) / diff * 255;
						if (dval > 64)
							has_white = true;
					} else {
						dval = (v.Get(x, y, d) - min) / diff * 255;
					}
					
					for (int dy = 0; dy < s; dy++) {
						RGBA* it2 = it + dy * W;
						for (int dx = 0; dx < s; dx++) {
							it2->r = dval;
							it2->g = dval;
							it2->b = dval;
							it2->a = 255;
							it2++;
						}
					}
				}
				// Color image
				else {
					byte r, g, b;
					if (draw_grads) {
						r = (v.GetGradient(x, y, d + 0) - min) / diff * 255;
						g = (v.GetGradient(x, y, d + 1) - min) / diff * 255;
						b = (v.GetGradient(x, y, d + 2) - min) / diff * 255;
						if (r > 64)
							has_white = true;
					} else {
						r = (v.Get(x, y, d + 0) - min) / diff * 255;
						g = (v.Get(x, y, d + 1) - min) / diff * 255;
						b = (v.Get(x, y, d + 2) - min) / diff * 255;
					}
					
					for (int dy = 0; dy < s; dy++) {
						RGBA* it2 = it + dy * W;
						for (int dx = 0; dx < s; dx++) {
							it2->r = r;
							it2->g = g;
							it2->b = b;
							it2->a = 255;
							it2++;
						}
					}
				}
				
				it += s;
			}
			it += (s - 1) * W;
			
			
		}
		
		// Gradient images need to be cached to drawer, because it does not fit in the engine.
		// This is not the correct solution, but it is the only solution I can think of currently.
		if (draw_grads) {
			int id = sz.cx * pt.y + pt.x;
			int i = gradient_cache.Find(id);
			if (has_white) {
				if (i == -1) {
					const Image& img = gradient_cache.Add(id, ib);
					draw.DrawImage(pt.x, pt.y, img);
				} else {
					const Image& img = (gradient_cache[i] = ib);
					draw.DrawImage(pt.x, pt.y, img);
				}
			} else {
				if (i == -1) {
					draw.DrawImage(pt.x, pt.y, ib);
				} else {
					const Image& img = gradient_cache[i];
					draw.DrawImage(pt.x, pt.y, img);
				}
			}
		}
		// Draw other images normally
		else {
			draw.DrawImage(pt.x, pt.y, ib);
		}
		
		if (pt.x + W > sz.cx) {
			pt.x = x;
			pt.y += H + 4;
		}
		else {
			pt.x += W + 4;
		}
		
	}
	
	// Next row
	if (end_newline) {
		pt.x = x;
		pt.y += H;
	}
}


















SessionConvLayers::SessionConvLayers() {
	ses = NULL;
	AddFrame(sb);
	sb.WhenScroll = THISBACK(Scroll);
	sb.SetLine(30);
	is_scrolling = false;
	is_color = false;
}

void SessionConvLayers::SetSession(Session& ses) {
	this->ses = &ses;
	ses.WhenSessionLoaded << THISBACK(RefreshLayers);
}

bool SessionConvLayers::Key(dword key, int) {
	return sb.VertKey(key);
}

void SessionConvLayers::MouseWheel(Point, int zdelta, dword) {
	sb.Wheel(zdelta);
}

void SessionConvLayers::Layout() {
	Size sz = GetSize();
	sb.SetPage(sz.cy);
	int scroll = sb;
	
	if (layer_ctrls.IsEmpty() || !ses) return;
	
	bool use_session = !is_scrolling;
	if (use_session) ses->Enter();
	
	Net& net = ses->GetNetwork();
	
	int y = -scroll;
	int total = 0;
	for(int i = 0; i < layer_ctrls.GetCount(); i++) {
		ConvLayerCtrl& ctrl = layer_ctrls[i];
		
		// Calculate height for current width
		// Make layout faster when scrolling by skipping size calculation
		if (use_session) {
			Size tmp_sz(sz.cx, 1);
			ImageDraw ib(tmp_sz);
			ctrl.ClearGradientCache();
			ctrl.PaintSize(ib, tmp_sz);
		}
		int h = ctrl.GetHeight();
		
		ctrl.SetRect(0, y, sz.cx, h);
		y += h;
		total += h;
	}
	
	// Avoid recursive layout callback
	sb.WhenScroll.Clear();
	sb.SetTotal(total);
	sb.WhenScroll = THISBACK(Scroll);
	
	if (use_session) ses->Leave();
}

void SessionConvLayers::Scroll() {
	is_scrolling = true;
	Layout();
	is_scrolling = false;
}

void SessionConvLayers::RefreshLayers() {
	for(int i = 0; i < layer_ctrls.GetCount(); i++) {
		RemoveChild(&layer_ctrls[i]);
	}
	layer_ctrls.Clear();
	
	ses->Enter();
	Net& net = ses->GetNetwork();
	
	int layer_count = net.GetLayers().GetCount();
	layer_ctrls.SetCount(layer_count);
	for(int i = 0; i < layer_ctrls.GetCount(); i++) {
		ConvLayerCtrl& ctrl = layer_ctrls[i];
		ctrl.SetSession(*ses);
		ctrl.SetId(i);
		ctrl.SetColor(is_color);
		ctrl.HideGradients(hide_gradients);
		Add(ctrl);
	}
	
	ses->Leave();
	
	PostCallback(THISBACK(Layout));
}

}


