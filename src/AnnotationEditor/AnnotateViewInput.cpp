#include "AnnotateView.h"

NAMESPACE_UPP

void AnnotateView::LeftDown(Point p, dword keyflags) {
	if(p.y < 30) return;
	if(current_tool == TOOL_REVIEW) {
		SetFocus(); Pointf pt_img = ScreenToImage(p);
		selected_id = -1; selected_ids.Clear(); selected_kp_id = -1;
		if(entry) {
			for(auto& obj : entry->suggestions) {
				if(!obj.visible) continue;
				if(!obj.bbox.Contains(pt_img)) continue;
				bool hit = false; for(const auto& poly : obj.polygons) if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
				if(hit) { selected_id = obj.id; selected_ids.Add(selected_id); SyncListSelectionFromCurrent(); Refresh(); return; }
			}
			for(int i = entry->annotations.GetCount() - 1; i >= 0; i--) {
				if(!entry->annotations[i].visible) continue;
				if(!entry->annotations[i].bbox.Contains(pt_img)) continue;
				bool hit = false; for(const auto& poly : entry->annotations[i].polygons) if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
				if(hit) { selected_id = entry->annotations[i].id; selected_ids.Add(selected_id); SyncListSelectionFromCurrent(); break; }
			}
		}
		Refresh();
		return;
	}
	SetFocus(); Pointf pt_img = ScreenToImage(p);
	if(current_tool == TOOL_BBOX) {
		if(active_category_id == -1) { PromptOK("Select a category first"); return; }
		if(!drawing_bbox) { drawing_bbox = true; bbox_start = pt_img; bbox_current = bbox_start; Refresh(); }
		else { drawing_bbox = false; AddBBox(bbox_start, pt_img); Refresh(); }
	} else if(current_tool == TOOL_POLYGON) {
		if(selected_id == -1) {
			if(chk_auto_create.Get() && active_category_id != -1) {
				CreateAutoObject();
			} else {
				PromptOK("Select an object in the tree first"); return;
			}
		}
		if(current_poly.GetCount() > 0 && DistSq(pt_img, current_poly[0]) < (10.0/zoom)*(10.0/zoom)) { ClosePolygon(); }
		else { current_poly.Add(pt_img); Refresh(); }
	} else if(current_tool == TOOL_BRUSH || current_tool == TOOL_ERASER) {
		if(selected_id == -1) { PromptOK("Select an object first"); return; }
		brushing = true; brush_strokes.Clear(); last_brush_pos = pt_img; brush_strokes.Add(CreateCirclePolygon(pt_img, brush_radius)); SetCapture(); Refresh();
	} else if(current_tool == TOOL_KEYPOINT) {
		if(selected_id == -1) { PromptOK("Select an object first"); return; }
		OnPlaceKeypoint(pt_img);
	} else if(current_tool == TOOL_MAGICWAND) {
		if(selected_id == -1) { PromptOK("Select an object first"); return; }
		OnMagicWand(p);
	} else if(current_tool == TOOL_SELECT) {
		bool multi_select_mod = (keyflags & K_CTRL) || (keyflags & K_SHIFT);
		if((keyflags & K_ALT) && entry) {
			if(selected_ids.IsEmpty() && selected_id != -1)
				selected_ids.Add(selected_id);
			move_obj_indices.Clear();
			move_old_polys.Clear();
			move_old_kps.Clear();
			for(int si = 0; si < selected_ids.GetCount(); si++) {
				int sid = selected_ids[si];
				for(int i = 0; i < entry->annotations.GetCount(); i++) {
					if(entry->annotations[i].id != sid) continue;
					move_obj_indices.Add(i);
					move_old_polys.Add() <<= entry->annotations[i].polygons;
					move_old_kps.Add() <<= entry->annotations[i].keypoints;
					break;
				}
			}
			if(!move_obj_indices.IsEmpty()) {
				move_start_img = pt_img;
				moving_objects = true;
				selected_kp_id = -1;
				SetCapture();
				return;
			}
		}
		if(entry && !multi_select_mod) {
			for(auto& obj : entry->annotations) {
				if(!obj.visible) continue;
				if(!obj.bbox.Contains(pt_img)) continue;
				for(auto& kp : obj.keypoints) {
					if(DistSq(pt_img, Pointf(kp.x, kp.y)) < (10.0/zoom)*(10.0/zoom)) {
						selected_kp_id = kp.id;
						selected_id = obj.id;
						selected_ids.Clear();
						selected_ids.Add(selected_id);
						dragging_kp = true;
						kp_old_pos = Pointf(kp.x, kp.y);
						SyncListSelectionFromCurrent();
						SetCapture();
						Refresh();
						return;
					}
				}
			}
		}
		if(selected_id != -1 && !multi_select_mod) {
			for(int i = 0; i < entry->annotations.GetCount(); i++) {
				if(entry->annotations[i].id == selected_id) {
					AnnotationObject& obj = entry->annotations[i]; if(!obj.visible) continue;
					if(!obj.bbox.Contains(pt_img)) continue;
					for(int j = 0; j < obj.polygons.GetCount(); j++) {
						for(int k = 0; k < obj.polygons[j].GetCount(); k++) {
							if(DistSq(pt_img, obj.polygons[j][k]) < (8.0/zoom)*(8.0/zoom)) {
								if(keyflags & K_SHIFT) { if(obj.polygons[j].GetCount() > 3) { if(cmdmgr) { cmdmgr->Execute(new DeleteVertexCommand(*entry, selected_id, j, k, obj.polygons[j][k])); RefreshAfterCommand(); } else { obj.polygons[j].Remove(k); obj.UpdateBBox(); Refresh(); } } return; }
								dragging_pt = true; drag_obj_idx = i; drag_poly_idx = j; drag_pt_idx = k; if(entry) drag_poly_before <<= entry->annotations[drag_obj_idx].polygons[drag_poly_idx];
								SetCapture(); return;
							}
						}
					}
				}
			}
		}
		int hit_suggestion_id = -1;
		int hit_annotation_id = -1;
		if(entry) {
			for(auto& obj : entry->suggestions) {
				if(!obj.visible) continue;
				if(!obj.bbox.Contains(pt_img)) continue;
				bool hit = false; for(const auto& poly : obj.polygons) if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
				if(hit) { hit_suggestion_id = obj.id; break; }
			}
			if(hit_suggestion_id == -1) {
				for(int i = entry->annotations.GetCount() - 1; i >= 0; i--) {
					if(!entry->annotations[i].visible) continue;
					if(!entry->annotations[i].bbox.Contains(pt_img)) continue;
					bool hit = false; for(const auto& poly : entry->annotations[i].polygons) if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
					if(hit) { hit_annotation_id = entry->annotations[i].id; break; }
				}
			}
		}
		selected_kp_id = -1;
		if(hit_suggestion_id != -1) {
			selected_id = hit_suggestion_id;
			selected_ids.Clear();
			selected_ids.Add(selected_id);
			SyncListSelectionFromCurrent();
			Refresh();
			return;
		}
		if(hit_annotation_id != -1) {
			if(multi_select_mod) {
				int fi = selected_ids.Find(hit_annotation_id);
				if(fi >= 0) selected_ids.Remove(fi);
				else selected_ids.Add(hit_annotation_id);
				selected_id = selected_ids.IsEmpty() ? -1 : selected_ids[selected_ids.GetCount() - 1];
			} else {
				selected_id = hit_annotation_id;
				selected_ids.Clear();
				selected_ids.Add(selected_id);
			}
			SyncListSelectionFromCurrent();
			Refresh();
			return;
		}
		selecting_rect = true;
		select_rect_start = pt_img;
		select_rect_current = pt_img;
		select_rect_toggle_mode = (keyflags & K_CTRL) != 0;
		select_rect_add_mode = (keyflags & K_SHIFT) != 0;
		selected_kp_id = -1;
		SetCapture();
		Refresh();
	}
}

void AnnotateView::LeftDouble(Point p, dword keyflags) {
	if(p.y < 30 || !entry)
		return;
	SetFocus();
	Pointf pt_img = ScreenToImage(p);
	selected_kp_id = -1;
	for(auto& obj : entry->annotations) {
		if(!obj.visible || !obj.bbox.Contains(pt_img))
			continue;
		for(auto& kp : obj.keypoints) {
			if(DistSq(pt_img, Pointf(kp.x, kp.y)) < (10.0 / zoom) * (10.0 / zoom)) {
				selected_id = obj.id;
				selected_ids.Clear();
				selected_ids.Add(selected_id);
				selected_kp_id = kp.id;
				SyncListSelectionFromCurrent();
				OnGeneralSettings();
				Refresh();
				return;
			}
		}
	}
	for(auto& obj : entry->suggestions) {
		if(!obj.visible || !obj.bbox.Contains(pt_img))
			continue;
		bool hit = false;
		for(const auto& poly : obj.polygons) if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
		if(hit) {
			selected_id = obj.id;
			selected_ids.Clear();
			selected_ids.Add(selected_id);
			SyncListSelectionFromCurrent();
			OnGeneralSettings();
			Refresh();
			return;
		}
	}
	for(int i = entry->annotations.GetCount() - 1; i >= 0; i--) {
		AnnotationObject& obj = entry->annotations[i];
		if(!obj.visible || !obj.bbox.Contains(pt_img))
			continue;
		bool hit = false;
		for(const auto& poly : obj.polygons) if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
		if(hit) {
			selected_id = obj.id;
			selected_ids.Clear();
			selected_ids.Add(selected_id);
			selected_kp_id = -1;
			SyncListSelectionFromCurrent();
			OnGeneralSettings();
			Refresh();
			return;
		}
	}
}

void AnnotateView::MouseMove(Point p, dword keyflags) {
	last_mouse_pos = p;
	mouse_pos_img = ScreenToImage(p);
	if(drawing_bbox) { bbox_current = mouse_pos_img; Refresh(); }
	if(selecting_rect) { select_rect_current = mouse_pos_img; Refresh(); return; }
	if(current_poly.GetCount() > 0) Refresh();
	if(panning) { offset.x = offset_start.x + (p.x - pan_start.x); offset.y = offset_start.y + (p.y - pan_start.y); Refresh(); }
	if(brushing) { if(DistSq(mouse_pos_img, last_brush_pos) > (brush_radius/2.0)*(brush_radius/2.0)) { brush_strokes.Add(CreateCirclePolygon(mouse_pos_img, brush_radius)); last_brush_pos = mouse_pos_img; Refresh(); } }
	if(moving_objects && entry) {
		Pointf delta(mouse_pos_img.x - move_start_img.x, mouse_pos_img.y - move_start_img.y);
		for(int i = 0; i < move_obj_indices.GetCount(); i++) {
			AnnotationObject& obj = entry->annotations[move_obj_indices[i]];
			obj.polygons <<= move_old_polys[i];
			obj.keypoints <<= move_old_kps[i];
			for(auto& poly : obj.polygons)
				for(auto& pt : poly) { pt.x += delta.x; pt.y += delta.y; }
			for(auto& kp : obj.keypoints) { kp.x += delta.x; kp.y += delta.y; }
			obj.UpdateBBox();
		}
		Refresh();
		return;
	}
	if(dragging_pt && entry) {
		AnnotationObject& obj = entry->annotations[drag_obj_idx]; Vector<Pointf>& poly = obj.polygons[drag_poly_idx]; Pointf new_pos = mouse_pos_img;
		if(IsRectangle(poly)) { int next = (drag_pt_idx + 1) % 4; int prev = (drag_pt_idx + 3) % 4; if(abs(poly[drag_pt_idx].x - poly[next].x) < 0.1) { poly[next].x = new_pos.x; poly[prev].y = new_pos.y; } else { poly[next].y = new_pos.y; poly[prev].x = new_pos.x; } poly[drag_pt_idx] = new_pos; }
		else poly[drag_pt_idx] = new_pos;
		obj.UpdateBBox();
		Refresh();
	}
	if(dragging_kp && entry) {
		for(auto& obj : entry->annotations) if(obj.id == selected_id) { for(auto& kp : obj.keypoints) if(kp.id == selected_kp_id) { kp.x = mouse_pos_img.x; kp.y = mouse_pos_img.y; break; } obj.UpdateBBox(); break; }
		Refresh();
	}
	int old_hov_id = hovered_id;
	int old_hov_kp_id = hovered_kp_id;
	hovered_id = -1; hovered_kp_id = -1;
	if(entry && !panning && !brushing && !dragging_pt && !dragging_kp) {
		Pointf pt_img = mouse_pos_img;
		for(auto& obj : entry->suggestions) {
			if(!obj.visible) continue;
			if(!obj.bbox.Contains(pt_img)) continue;
			bool hit = false; for(const auto& poly : obj.polygons) if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
			if(hit) { hovered_id = obj.id; goto hover_done; }
		}
		for(auto& obj : entry->annotations) {
			if(!obj.visible) continue;
			if(!obj.bbox.Contains(pt_img)) continue;
			for(auto& kp : obj.keypoints) {
				if(DistSq(pt_img, Pointf(kp.x, kp.y)) < (10.0/zoom)*(10.0/zoom)) {
					hovered_kp_id = kp.id;
					goto hover_done;
				}
			}
		}
		for(int i = entry->annotations.GetCount() - 1; i >= 0; i--) {
			if(!entry->annotations[i].visible) continue;
			if(!entry->annotations[i].bbox.Contains(pt_img)) continue;
			bool hit = false; for(const auto& poly : entry->annotations[i].polygons) if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
			if(hit) { hovered_id = entry->annotations[i].id; break; }
		}
	}
hover_done:
	if(hovered_id != old_hov_id || hovered_kp_id != old_hov_kp_id) Refresh();
}

void AnnotateView::LeftUp(Point p, dword keyflags) {
	if(selecting_rect) {
		selecting_rect = false;
		ReleaseCapture();
		Pointf a = select_rect_start;
		Pointf b = select_rect_current;
		double min_x = min(a.x, b.x), max_x = max(a.x, b.x);
		double min_y = min(a.y, b.y), max_y = max(a.y, b.y);
		double min_drag = max(2.0 / max(zoom, 0.001), 0.5);
		bool tiny = (max_x - min_x) < min_drag && (max_y - min_y) < min_drag;

		Vector<int> hits;
		if(!tiny && entry) {
			for(int i = entry->annotations.GetCount() - 1; i >= 0; i--) {
				const auto& obj = entry->annotations[i];
				if(!obj.visible) continue;
				if(obj.bbox.right < min_x || obj.bbox.left > max_x ||
				   obj.bbox.bottom < min_y || obj.bbox.top > max_y)
					continue;
				hits.Add(obj.id);
			}
		}

		if(select_rect_toggle_mode) {
			for(int i = 0; i < hits.GetCount(); i++) {
				int id = hits[i];
				int fi = selected_ids.Find(id);
				if(fi >= 0) selected_ids.Remove(fi);
				else selected_ids.Add(id);
			}
		} else if(select_rect_add_mode) {
			for(int i = 0; i < hits.GetCount(); i++) {
				int id = hits[i];
				if(selected_ids.Find(id) < 0) selected_ids.Add(id);
			}
		} else {
			selected_ids.Clear();
			if(!tiny) {
				for(int i = hits.GetCount() - 1; i >= 0; i--)
					selected_ids.Add(hits[i]);
			}
		}

		selected_id = selected_ids.IsEmpty() ? -1 : selected_ids.Top();
		selected_kp_id = -1;
		SyncListSelectionFromCurrent();
		Refresh();
		return;
	}
	if(moving_objects && entry) {
		moving_objects = false;
		ReleaseCapture();
		Vector<Vector<Vector<Pointf>>> new_polys;
		Vector<Vector<KeypointInstance>> new_kps;
		for(int i = 0; i < move_obj_indices.GetCount(); i++) {
			const AnnotationObject& obj = entry->annotations[move_obj_indices[i]];
			new_polys.Add() <<= obj.polygons;
			new_kps.Add() <<= obj.keypoints;
		}
		if(cmdmgr && !move_obj_indices.IsEmpty()) {
			cmdmgr->Execute(new MoveObjectsCommand(*entry, move_obj_indices, move_old_polys, move_old_kps, new_polys, new_kps));
			RefreshAfterCommand();
		} else {
			RefreshObjectTree();
			Refresh();
			WhenDirty();
		}
		move_obj_indices.Clear();
		move_old_polys.Clear();
		move_old_kps.Clear();
		return;
	}
	if(brushing && entry) {
		brushing = false; ReleaseCapture(); AnnotationObject* obj = nullptr; for(auto& o : entry->annotations) if(o.id == selected_id) { obj = &o; break; }
		if(obj) {
			Vector<Vector<Pointf>> new_polys; if(current_tool == TOOL_ERASER) new_polys = SubtractPolygons(obj->polygons, brush_strokes); else new_polys = UnionPolygons(obj->polygons, brush_strokes);
			if(cmdmgr) { cmdmgr->Execute(new BrushEditCommand(*entry, obj->id, obj->polygons, new_polys, current_tool == TOOL_ERASER)); RefreshAfterCommand(); }
			else { obj->polygons <<= new_polys; obj->UpdateBBox(); Refresh(); }
		}
		brush_strokes.Clear();
	}
	if(dragging_pt) { dragging_pt = false; ReleaseCapture(); if(cmdmgr && entry) { AnnotationObject& obj = entry->annotations[drag_obj_idx]; cmdmgr->Execute(new MoveVertexCommand(*entry, obj.id, drag_poly_idx, drag_poly_before, obj.polygons[drag_poly_idx])); RefreshAfterCommand(); } else Refresh(); drag_poly_before.Clear(); }
	if(dragging_kp) { dragging_kp = false; ReleaseCapture(); if(cmdmgr && entry) { cmdmgr->Execute(new MoveKeypointCommand(*entry, selected_id, selected_kp_id, kp_old_pos, mouse_pos_img)); RefreshAfterCommand(); } else Refresh(); }
}

bool AnnotateView::CancelActiveInteractions() {
	bool changed = false;
	if(drawing_bbox) { drawing_bbox = false; changed = true; }
	if(selecting_rect) { selecting_rect = false; changed = true; }
	if(!current_poly.IsEmpty()) { current_poly.Clear(); changed = true; }
	if(brushing) { brushing = false; brush_strokes.Clear(); changed = true; }
	if(dragging_pt) { dragging_pt = false; drag_poly_before.Clear(); changed = true; }
	if(dragging_kp) { dragging_kp = false; changed = true; }
	if(moving_objects) {
		moving_objects = false;
		if(entry) {
			for(int i = 0; i < move_obj_indices.GetCount() && i < move_old_polys.GetCount() && i < move_old_kps.GetCount(); i++) {
				AnnotationObject& obj = entry->annotations[move_obj_indices[i]];
				obj.polygons <<= move_old_polys[i];
				obj.keypoints <<= move_old_kps[i];
				obj.UpdateBBox();
			}
		}
		move_obj_indices.Clear();
		move_old_polys.Clear();
		move_old_kps.Clear();
		changed = true;
	}
	if(changed) {
		ReleaseCapture();
		RefreshObjectTree();
		Refresh();
	}
	return changed;
}

void AnnotateView::MiddleDown(Point p, dword keyflags) {
	if(p.y < 30) return;
	panning = true;
	pan_start = p;
	offset_start = offset;
	SetCapture();
}

void AnnotateView::MiddleUp(Point p, dword keyflags) {
	panning = false;
	ReleaseCapture();
}

void AnnotateView::RightDown(Point p, dword keyflags) {
	if(p.y < 30) return;
	CancelActiveInteractions();
	if(!entry) return;

	Pointf pt_img = ScreenToImage(p);
	int hit_annotation_id = -1;
	for(int i = entry->annotations.GetCount() - 1; i >= 0; i--) {
		const AnnotationObject& obj = entry->annotations[i];
		if(!obj.visible) continue;
		if(!obj.bbox.Contains(pt_img)) continue;
		bool hit = false;
		for(const auto& poly : obj.polygons)
			if(IsPointInPolygon(pt_img, poly)) { hit = true; break; }
		if(hit) { hit_annotation_id = obj.id; break; }
	}
	if(hit_annotation_id == -1) return;

	selected_id = hit_annotation_id;
	selected_ids.Clear();
	selected_ids.Add(selected_id);
	selected_kp_id = -1;
	SyncListSelectionFromCurrent();
	Refresh();

	bool has_slot_id = false;
	for(const auto& obj : entry->annotations) {
		if(obj.id != selected_id) continue;
		String sid = obj.metadata.Get(MluiSlotIdKey(), "");
		if(sid.IsEmpty()) sid = obj.slot_id;
		has_slot_id = !sid.IsEmpty();
		break;
	}

	MenuBar menu;
	menu.Add("Copy hint to last MLUI script", [=] {
		if(!WhenCopyHintToLastMluiScript || !entry || selected_id == -1) return;
		for(const auto& obj : entry->annotations) {
			if(obj.id == selected_id) {
				WhenCopyHintToLastMluiScript(obj);
				break;
			}
		}
	}).Enable(has_slot_id && (bool)WhenCopyHintToLastMluiScript);
	menu.Separator();
	menu.Add("Settings...", THISBACK(OnObjectSettings));
	menu.Add("Set Geometry...", THISBACK(OnSetGeometry));
	menu.Add("Delete", THISBACK(OnObjectDelete));
	menu.Execute();
}

void AnnotateView::MouseWheel(Point p, int zdelta, dword keyflags) {
	if(p.y < 30) return;
	double old_zoom = zoom;
	if(zdelta > 0) zoom *= 1.2; else zoom /= 1.2;
	if(zoom < 0.05) zoom = 0.05;
	if(zoom > 50.0) zoom = 50.0;
	offset.x = p.x - (p.x - offset.x) * (zoom / old_zoom);
	offset.y = p.y - (p.y - offset.y) * (zoom / old_zoom);
	InvalidateScaledImage();
	Refresh();
}

END_UPP_NAMESPACE
