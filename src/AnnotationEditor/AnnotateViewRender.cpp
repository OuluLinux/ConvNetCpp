#include "AnnotateView.h"

NAMESPACE_UPP

void AnnotateView::CenterImage() {
	if(!img) return;
	Size sz = GetSize();
	if(sz.cx <= 0) sz.cx = 1;
	sz.cy = max(1, sz.cy - 30);
	double ratioX = img.GetWidth() > 0 ? (double)sz.cx / img.GetWidth() : zoom;
	double ratioY = img.GetHeight() > 0 ? (double)sz.cy / img.GetHeight() : zoom;
	double target_ratio = min(ratioX, ratioY);
	if(target_ratio <= 0) target_ratio = 1.0;
	zoom = min(max(target_ratio * 0.9, 0.05), 20.0);
	offset.x = (sz.cx - img.GetWidth() * zoom) / 2.0;
	offset.y = (sz.cy - img.GetHeight() * zoom) / 2.0 + 30;
	InvalidateScaledImage();
	Refresh();
}

Rectf AnnotateView::GetViewportRect() {
	Size sz = GetSize();
	Pointf tl = ScreenToImage(Point(0, 30));
	Pointf br = ScreenToImage(Point(sz.cx, sz.cy));
	return Rectf(tl.x, tl.y, br.x, br.y);
}

void AnnotateView::UpdateToolLabel() {
	String text = "Tool: ";
	switch(current_tool) {
		case TOOL_SELECT: text << "Select"; break;
		case TOOL_BBOX: text << "BBox"; break;
		case TOOL_POLYGON: text << "Polygon"; break;
		case TOOL_BRUSH: text << "Brush"; break;
		case TOOL_ERASER: text << "Eraser"; break;
		case TOOL_KEYPOINT: text << "Keypoint"; break;
		case TOOL_MAGICWAND: text << "Wand"; break;
		case TOOL_REVIEW: text << "Review"; break;
	}
	lbl_active_tool.SetLabel(text);
}

void AnnotateView::InvalidateScaledImage() {
	scaled_img.Clear();
	scaled_size = Size(0, 0);
	scaled_zoom = -1;
}

void AnnotateView::EnsureScaledImage() {
	if(!img) {
		InvalidateScaledImage();
		return;
	}
	const int MAX_RENDER_PIXELS = 6000;
	int desired_cx = min(MAX_RENDER_PIXELS, max(1, int(img.GetWidth() * zoom)));
	int desired_cy = min(MAX_RENDER_PIXELS, max(1, int(img.GetHeight() * zoom)));
	if(desired_cx != scaled_size.cx || desired_cy != scaled_size.cy || fabs(zoom - scaled_zoom) > 1e-6) {
		scaled_img = Rescale(img, Size(desired_cx, desired_cy));
		scaled_size = Size(desired_cx, desired_cy);
		scaled_zoom = zoom;
	}
}

void AnnotateView::Paint(Draw& w) {
	UpdateToolLabel();
	Size sz = GetSize(); w.DrawRect(sz, SColorPaper()); if(!img) return;
	w.Clip(0, 30, sz.cx, max(0, sz.cy - 30));
	EnsureScaledImage();
	w.DrawImage(int(offset.x), int(offset.y), scaled_size.cx, scaled_size.cy, scaled_img);
	DrawAnnotations(w);

	String hint;
	if(!entry || entry->annotations.IsEmpty()) {
		hint = "Hint: Press B to create first object";
	} else if(selected_id != -1) {
		AnnotationObject* obj = nullptr; for(auto& o : entry->annotations) if(o.id == selected_id) { obj = &o; break; }
		if(obj && obj->keypoints.IsEmpty()) {
			Category* cat = nullptr; for(auto& c : *categories) if(c.id == obj->category_id) cat = &c;
			if(cat && !cat->keypoint_labels.IsEmpty()) hint = "Suggest: Press K to add keypoints";
		}
	}

	if(!hint.IsEmpty()) {
		Size hsz = GetTextSize(hint, StdFont().Italic());
		w.DrawText((sz.cx - hsz.cx) / 2, 40, hint, StdFont().Italic(), Gray());
	}

	if(drawing_bbox) {
		Point p1 = ImageToScreen(bbox_start); Point p2 = ImageToScreen(bbox_current);
		Rect r(min(p1.x, p2.x), min(p1.y, p2.y), max(p1.x, p2.x), max(p1.y, p2.y));
		w.DrawRect(r.left, r.top, r.Width(), 1, Black()); w.DrawRect(r.left, r.bottom, r.Width(), 1, Black());
		w.DrawRect(r.left, r.top, 1, r.Height(), Black()); w.DrawRect(r.right, r.top, 1, r.Height() + 1, Black());
	}
	if(selecting_rect) {
		Point p1 = ImageToScreen(select_rect_start);
		Point p2 = ImageToScreen(select_rect_current);
		Rect r(min(p1.x, p2.x), min(p1.y, p2.y), max(p1.x, p2.x), max(p1.y, p2.y));
		w.DrawRect(r.left, r.top, r.Width(), 1, LtBlue());
		w.DrawRect(r.left, r.bottom, r.Width(), 1, LtBlue());
		w.DrawRect(r.left, r.top, 1, r.Height(), LtBlue());
		w.DrawRect(r.right, r.top, 1, r.Height() + 1, LtBlue());
	}
	if(current_poly.GetCount() > 0) {
		Point p_last = ImageToScreen(current_poly.Top()); Point p_cur = ImageToScreen(mouse_pos_img);
		for(int i = 0; i < current_poly.GetCount() - 1; i++) w.DrawLine(ImageToScreen(current_poly[i]), ImageToScreen(current_poly[i+1]), 1, Black());
		w.DrawLine(p_last, p_cur, 1, Black());
		Point p_first = ImageToScreen(current_poly[0]); w.DrawEllipse(p_first.x - 4, p_first.y - 4, 8, 8, White(), 1, Black());
	}
	if(brushing) {
		for(const auto& stroke : brush_strokes) {
			for(int i = 0; i < stroke.GetCount(); i++)
				w.DrawLine(ImageToScreen(stroke[i]), ImageToScreen(stroke[(i+1)%stroke.GetCount()]), 1, Black());
		}
	}

	if(!temp_overlay_text.IsEmpty()) {
		Size tsz = GetTextSize(temp_overlay_text, StdFont(20).Bold());
		int tx = (sz.cx - tsz.cx) / 2;
		int ty = sz.cy - 100;
		w.DrawRect(tx - 10, ty - 5, tsz.cx + 20, tsz.cy + 10, Color(0, 0, 0));
		w.DrawText(tx, ty, temp_overlay_text, StdFont(20).Bold(), White());
		if(temp_overlay_timeout > 0) {
			temp_overlay_timeout--;
			if(temp_overlay_timeout == 0) temp_overlay_text = "";
			Refresh();
		}
	}

	w.End();
}

void AnnotateView::DrawAnnotations(Draw& w) {
	if(!entry) return;
	Rectf view = GetViewportRect();
	bool review = (current_tool == TOOL_REVIEW);

	auto ResolveColor = [&](const AnnotationObject& obj) -> Color {
		if(!IsNull(obj.color)) return obj.color;
		if(categories) {
			for(const auto& c : *categories)
				if(c.id == obj.category_id && !IsNull(c.color)) return c.color;
		}
		return Color(160, 160, 160);
	};

	auto DrawScreenRect = [&](const Rect& ir, Color c, int thick) {
		Point p1 = ImageToScreen(Pointf(ir.left, ir.top));
		Point p2 = ImageToScreen(Pointf(ir.right, ir.bottom));
		Rect r(min(p1.x, p2.x), min(p1.y, p2.y), max(p1.x, p2.x), max(p1.y, p2.y));
		for(int t = 0; t < max(1, thick); t++) {
			w.DrawRect(r.left + t, r.top + t, max(1, r.Width() - 2 * t), 1, c);
			w.DrawRect(r.left + t, r.bottom - 1 - t, max(1, r.Width() - 2 * t), 1, c);
			w.DrawRect(r.left + t, r.top + t, 1, max(1, r.Height() - 2 * t), c);
			w.DrawRect(r.right - 1 - t, r.top + t, 1, max(1, r.Height() - 2 * t), c);
		}
	};

	auto DrawSubLayoutOverlay = [&](const AnnotationObject& obj, bool is_selected, bool is_hovered) {
		if(!is_selected && !is_hovered)
			return;
		String sub_json = TrimBoth(obj.metadata.Get("sub_layout_json", ""));
		if(sub_json.IsEmpty())
			sub_json = TrimBoth(obj.metadata.Get("annlay_sub_layout_json", ""));
		if(sub_json.IsEmpty())
			return;

		Rect base((int)floor(obj.bbox.left), (int)floor(obj.bbox.top),
		          (int)ceil(obj.bbox.right), (int)ceil(obj.bbox.bottom));
		if(base.IsEmpty())
			return;

		AnnLaySlot slot;
		slot.sub_layout_json = sub_json;
		VectorMap<String, Rectf> regions;
		if(!AnnLayTryGetSubLayoutRegions(slot, regions))
			return;

		struct RegionDrawDef {
			const char* key;
			const char* label;
			Color color;
		};
		static const RegionDrawDef defs[] = {
			{"card_region", "element", Color(255, 210, 0)},
			{"rank_region", "level", Color(0, 220, 255)},
			{"suit_region", "category", Color(255, 120, 80)},
		};

		for(const RegionDrawDef& d : defs) {
			Rect rr = AnnLayResolveRegionRect(slot, base, d.key);
			if(rr.IsEmpty())
				continue;
			DrawScreenRect(rr, d.color, is_selected ? 2 : 1);
			Point tp = ImageToScreen(Pointf(rr.left, rr.top));
			w.DrawText(tp.x + 2, tp.y + 2, d.label, StdFont(8).Bold(), d.color);
		}

		double rot = 0;
		if(AnnLayTryGetSubLayoutRotationDeg(slot, rot)) {
			Point rp = ImageToScreen(Pointf(base.left, base.bottom));
			w.DrawText(rp.x + 2, rp.y - 12, Format("rot=%.1f", rot), StdFont(8).Bold(), Color(220, 220, 220));
		}
	};

	auto DrawObjs = [&](const Vector<AnnotationObject>& objs, bool is_sug) {
		for(int i = 0; i < objs.GetCount(); i++) {
			const AnnotationObject& obj = objs[i];
			if(!obj.visible) continue;
			if(obj.bbox.right < view.left || obj.bbox.left > view.right ||
			   obj.bbox.bottom < view.top || obj.bbox.top > view.bottom) continue;

			bool is_selected = (selected_ids.Find(obj.id) >= 0) || (selected_ids.IsEmpty() && selected_id == obj.id);
			bool is_hovered = (hovered_id == obj.id);

			Color color = is_sug ? Yellow() : ResolveColor(obj);
			if(obj.label_visibility_state == 0) color = Color(color.GetR()/2, color.GetG()/2, color.GetB()/2);
			if(!is_selected && !is_sug) color = Color(color.GetR()/2, color.GetG()/2, color.GetB()/2);
			if(is_selected) color = LtBlue();
			if(is_hovered) color = Color(min(255, color.GetR() + 60), min(255, color.GetG() + 60), min(255, color.GetB() + 60));
			if(review && !is_selected) color = Color(color.GetR()/4, color.GetG()/4, color.GetB()/4);

			int thick = is_selected ? 3 : (is_hovered ? 2 : 1);
			for(int p_idx = 0; p_idx < obj.polygons.GetCount(); p_idx++) {
				const auto& poly = obj.polygons[p_idx]; if(poly.GetCount() < 2) continue;
				if(is_sug) {
					for(int j = 0; j < poly.GetCount(); j++) {
						Point p1 = ImageToScreen(poly[j]); Point p2 = ImageToScreen(poly[(j + 1) % poly.GetCount()]);
						DrawDashLine(w, p1, p2, thick, color);
					}
				} else {
					for(int j = 0; j < poly.GetCount(); j++) {
						Point p1 = ImageToScreen(poly[j]); Point p2 = ImageToScreen(poly[(j + 1) % poly.GetCount()]);
						w.DrawLine(p1, p2, thick, color);
					}
				}
				if(is_selected && current_tool == TOOL_SELECT) for(int j = 0; j < poly.GetCount(); j++) { Point p = ImageToScreen(poly[j]); Color c = White(); w.DrawEllipse(p.x - 3, p.y - 3, 6, 6, c, 1, Black()); }
			}

			if(obj.label_visibility_state == 2) {
				Point p = ImageToScreen(Pointf(obj.bbox.left, obj.bbox.top));
				w.DrawText(p.x, p.y - 15, obj.name, StdFont(10).Bold(), SColorText());
			}

			for(int j = 0; j < obj.keypoints.GetCount(); j++) {
				const auto& kp = obj.keypoints[j]; Point p = ImageToScreen(Pointf(kp.x, kp.y));
				bool kp_sel = (selected_kp_id == kp.id);
				bool kp_hov = (hovered_kp_id == kp.id);
				int r = kp_sel ? 6 : (kp_hov ? 5 : 4);
				Color kp_color = is_sug ? Yellow() : obj.color;
				if(kp.visibility_state == 0) kp_color = Gray();
				w.DrawEllipse(p.x - r, p.y - r, 2 * r, 2 * r, kp_color, 1, Black());
				if(kp_sel) w.DrawEllipse(p.x - r - 2, p.y - r - 2, 2 * r + 4, 2 * r + 4, Null, 1, White());
				if(kp.visibility_state == 2) w.DrawText(p.x + r + 2, p.y - r, kp.label, StdFont(8), SColorText());
			}

			if(!is_sug)
				DrawSubLayoutOverlay(obj, is_selected, is_hovered);
		}
	};

	DrawObjs(entry->suggestions, true);
	DrawObjs(entry->annotations, false);

	if(show_hover_info && (hovered_id != -1 || hovered_kp_id != -1)) {
		DrawHoverInfo(w);
	}
}

void AnnotateView::DrawDashLine(Draw& w, Point p1, Point p2, int thick, Color c) {
	double dist = sqrt(DistSq(p1, p2));
	if(dist < 1) return;
	for(double d = 0; d < dist; d += 10) {
		double t1 = d / dist;
		double t2 = min(d + 5, dist) / dist;
		w.DrawLine(Point(int(p1.x + t1*(p2.x - p1.x)), int(p1.y + t1*(p2.y - p1.y))),
		           Point(int(p1.x + t2*(p2.x - p1.x)), int(p1.y + t2*(p2.y - p1.y))), thick, c);
	}
}

double AnnotateView::GetCategoryAcceptanceRate(int cat_id) {
	if(!dataset) return 0;
	int total = 0;
	int accepted = 0;
	for(const auto& ds : *datasets_ptr) {
		for(const auto& ie : ds.images) {
			for(const auto& obj : ie.annotations) {
				if(obj.category_id == cat_id && obj.accepted) { total++; accepted++; }
			}
			for(const auto& obj : ie.rejected_suggestions) {
				if(obj.category_id == cat_id) { total++; }
			}
		}
	}
	return total > 0 ? (double)accepted * 100.0 / total : 0;
}

void AnnotateView::DrawHoverInfo(Draw& w) {
	String info;
	if(hovered_kp_id != -1) {
		for(const auto& obj : entry->annotations) {
			for(const auto& kp : obj.keypoints) {
				if(kp.id == hovered_kp_id) {
					info << "Keypoint: " << kp.label << "\n";
					info << "State: " << GetStateText(kp.visibility_state) << "\n";
					info << "Object: " << obj.name << "\n";
					Category* cat = nullptr; for(auto& c : *categories) if(c.id == obj.category_id) cat = &c;
					if(cat) info << "Category: " << cat->name << "\n";
					info << "ID: " << kp.id;
					goto found;
				}
			}
		}
	} else if(hovered_id != -1) {
		for(const auto& obj : entry->annotations) {
			if(obj.id == hovered_id) {
				info << "Object: " << obj.name << "\n";
				Category* cat = nullptr; for(auto& c : *categories) if(c.id == obj.category_id) cat = &c;
				if(cat) info << "Category: " << cat->name << "\n";
				info << "ID: " << obj.id << "\n";
				info << "Label: " << GetStateText(obj.label_visibility_state) << "\n";

				int vc = 0; for(const auto& poly : obj.polygons) vc += poly.GetCount();
				info << "Geometry: " << obj.polygons.GetCount() << " polys, " << vc << " verts\n";
				info << "Keypoints: " << obj.keypoints.GetCount() << "\n";
				if(IsSuggestion(obj.id)) info << "Confidence: " << Format("%.2f", obj.score) << "\n";

				if(obj.metadata.IsEmpty()) info << "No metadata";
				else {
					info << "Metadata:\n";
					for(int i = 0; i < min(3, obj.metadata.GetCount()); i++)
						info << "  " << obj.metadata.GetKey(i) << ": " << obj.metadata[i] << "\n";
					if(obj.metadata.GetCount() > 3) info << "  ... (more)";
				}
				break;
			}
		}
	}
found:
	if(info.IsEmpty()) return;

	Size isz = GetTextSize(info, StdFont());
	isz.cx += 10; isz.cy += 10;
	Point p = last_mouse_pos;
	p.x += 15; p.y += 15;

	w.DrawRect(p.x, p.y, isz.cx, isz.cy, White());
	w.DrawRect(p.x, p.y, isz.cx, 1, Black());
	w.DrawRect(p.x, p.y + isz.cy - 1, isz.cx, 1, Black());
	w.DrawRect(p.x, p.y, 1, isz.cy, Black());
	w.DrawRect(p.x + isz.cx - 1, p.y, 1, isz.cy, Black());

	w.DrawText(p.x + 5, p.y + 5, info, StdFont(), Black());
}

END_UPP_NAMESPACE
