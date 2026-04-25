#include "AnnotateView.h"

NAMESPACE_UPP

void AnnotateView::ClosePolygon() {
	if(current_poly.GetCount() >= 3 && selected_id != -1) {
		if(cmdmgr) { cmdmgr->Execute(new AddPolygonCommand(*entry, selected_id, current_poly)); RefreshAfterCommand(); }
		else { for(int i = 0; i < entry->annotations.GetCount(); i++) if(entry->annotations[i].id == selected_id) { entry->annotations[i].polygons.Add() <<= current_poly; entry->annotations[i].UpdateBBox(); entry->has_annotations = true; break; } }
	}
	current_poly.Clear(); Refresh();
}

void AnnotateView::CreateAutoObject() {
	if(!entry || active_category_id == -1) return;
	AnnotationObject obj; static int next_id = 5000; obj.id = next_id++; obj.category_id = active_category_id;
	String cat_name = "Unknown"; Color cat_color = Black(); for(auto& c : *categories) if(c.id == active_category_id) { cat_name = c.name; cat_color = c.color; break; }
	obj.name = cat_name + " " + AsString(entry->annotations.GetCount() + 1); obj.color = cat_color;
	obj.UpdateBBox();
	if(cmdmgr) { cmdmgr->Execute(new CreateObjectCommand(*entry, obj)); selected_id = obj.id; selected_ids.Clear(); selected_ids.Add(selected_id); RefreshAfterCommand(); }
	else { entry->annotations.Add(pick(obj)); selected_id = entry->annotations.Top().id; selected_ids.Clear(); selected_ids.Add(selected_id); RefreshObjectTree(); Refresh(); }
}

void AnnotateView::AddBBox(Pointf start, Pointf end) {
	if(!entry || !categories) return; double min_x = min(start.x, end.x); double max_x = max(start.x, end.x); double min_y = min(start.y, end.y); double max_y = max(start.y, end.y); if(max_x - min_x < 2 || max_y - min_y < 2) return;
	AnnotationObject obj; static int next_id = 1000; obj.id = next_id++; obj.category_id = active_category_id;
	String cat_name = "Unknown"; Color cat_color = Black(); for(int i = 0; i < categories->GetCount(); i++) if((*categories)[i].id == active_category_id) { cat_name = (*categories)[i].name; cat_color = (*categories)[i].color; break; }
	obj.name = cat_name + " " + AsString(entry->annotations.GetCount() + 1); obj.color = cat_color; Vector<Pointf>& poly = obj.polygons.Add(); poly.Add(Pointf(min_x, min_y)); poly.Add(Pointf(max_x, min_y)); poly.Add(Pointf(max_x, max_y)); poly.Add(Pointf(min_x, max_y));
	obj.UpdateBBox();
	if(cmdmgr) { cmdmgr->Execute(new CreateObjectCommand(*entry, obj)); selected_id = obj.id; selected_ids.Clear(); selected_ids.Add(selected_id); RefreshAfterCommand(); }
	else { entry->annotations.Add(pick(obj)); entry->has_annotations = true; selected_id = entry->annotations.Top().id; selected_ids.Clear(); selected_ids.Add(selected_id); RefreshObjectTree(); Refresh(); }
	if(WhenBBoxAdded) WhenBBoxAdded(selected_id);
}

void AnnotateView::OnPlaceKeypoint(Pointf pt) {
	if(!entry || selected_id == -1) return;
	AnnotationObject* obj = nullptr; for(auto& o : entry->annotations) if(o.id == selected_id) { obj = &o; break; }
	if(!obj) return;
	Category* cat = nullptr; for(auto& c : *categories) if(c.id == obj->category_id) { cat = &c; break; }
	if(!cat || cat->keypoint_labels.IsEmpty()) { PromptOK("Selected category has no keypoint definitions"); return; }
	String label;
	for(const auto& l : cat->keypoint_labels) {
		bool used = false; for(const auto& k : obj->keypoints) if(k.label == l) { used = true; break; }
		if(!used) { label = l; break; }
	}
	if(label.IsEmpty()) { PromptOK("All keypoints already used for this object"); return; }
	KeypointInstance kp; static int next_kp_id = 5000; kp.id = next_kp_id++; kp.label = label; kp.x = pt.x; kp.y = pt.y; kp.visibility_state = 2;
	if(cmdmgr) { cmdmgr->Execute(new AddKeypointCommand(*entry, obj->id, kp)); RefreshAfterCommand(); }
	else { obj->keypoints.Add(kp); obj->UpdateBBox(); RefreshObjectTree(); Refresh(); }
}

void AnnotateView::OnMagicWand(Point p) {
	if(!entry || !img || selected_id == -1) return;
	Point pt_img(int((p.x - offset.x) / zoom), int((p.y - offset.y) / zoom));
	Vector<Vector<Pointf>> wand_polys = MagicWand(img, pt_img, ~edit_threshold, 0);
	if(wand_polys.IsEmpty()) { WhenLog("Magic Wand found empty region"); return; }
	AnnotationObject* obj = nullptr; for(auto& o : entry->annotations) if(o.id == selected_id) { obj = &o; break; }
	if(obj) {
		Vector<Vector<Pointf>> new_polys = UnionPolygons(obj->polygons, wand_polys);
		if(cmdmgr) { cmdmgr->Execute(new BrushEditCommand(*entry, obj->id, obj->polygons, new_polys, false)); RefreshAfterCommand(); }
		else { obj->polygons <<= new_polys; obj->UpdateBBox(); Refresh(); }
		WhenLog(Format("Magic Wand applied to object %d", selected_id));
	}
}

void AnnotateView::DuplicateObject() {
	if(selected_id == -1 || !entry) return;
	for(const auto& o : entry->annotations) {
		if(o.id == selected_id) {
			AnnotationObject copy(o);
			static int next_id = 3000; copy.id = next_id++;
			copy.name << " (Copy)";
			if(cmdmgr) {
				cmdmgr->Execute(new CreateObjectCommand(*entry, copy));
				selected_id = copy.id;
				selected_ids.Clear();
				selected_ids.Add(selected_id);
				RefreshAfterCommand();
			} else {
				entry->annotations.Add(pick(copy));
				selected_id = entry->annotations.Top().id;
				selected_ids.Clear();
				selected_ids.Add(selected_id);
				RefreshObjectTree(); Refresh();
			}
			WhenLog(Format("Duplicated object %d", o.id));
			break;
		}
	}
}

void AnnotateView::CopyAnnotations() {
	if(selected_id == -1 || !entry) return;
	for(const auto& o : entry->annotations) {
		if(o.id == selected_id) {
			clipboard_object = o;
			WhenLog(Format("Copied object %d to clipboard", o.id));
			break;
		}
	}
}

void AnnotateView::PasteAnnotations() {
	if(clipboard_object.id == -1 || !entry) return;
	AnnotationObject paste(clipboard_object);
	static int next_id = 4000; paste.id = next_id++;
	if(cmdmgr) {
		cmdmgr->Execute(new CreateObjectCommand(*entry, paste));
		selected_id = paste.id;
		selected_ids.Clear();
		selected_ids.Add(selected_id);
		RefreshAfterCommand();
	} else {
		entry->annotations.Add(pick(paste));
		selected_id = entry->annotations.Top().id;
		selected_ids.Clear();
		selected_ids.Add(selected_id);
		RefreshObjectTree(); Refresh();
	}
	WhenLog("Pasted object from clipboard");
}

void AnnotateView::CycleVisibility() {
	if(selected_id == -1 || !entry) return;
	for(auto& o : entry->annotations) {
		if(o.id == selected_id) {
			o.label_visibility_state = (o.label_visibility_state + 1) % 3;
			RefreshObjectTree(); Refresh(); WhenDirty();
			WhenLog(Format("Cycled visibility for object %d: %s", o.id, GetStateText(o.label_visibility_state)));
			break;
		}
	}
}

void AnnotateView::SelectCategory(int index) {
	if(!categories || index < 0 || index >= categories->GetCount()) return;
	active_category_id = (*categories)[index].id;
	cat_list.SetCursor(index);
	WhenLog("Selected category: " + (*categories)[index].name);
	temp_overlay_text = "Category: " + (*categories)[index].name;
	temp_overlay_timeout = 60;
	Refresh();
}

void AnnotateView::SetActiveCategoryById(int category_id) {
	active_category_id = category_id;
	if(!categories) return;
	for(int i = 0; i < categories->GetCount(); i++) {
		if((*categories)[i].id == category_id) {
			cat_list.SetCursor(i);
			return;
		}
	}
}

void AnnotateView::SetCurrentTool(ToolType tool) {
	current_tool = tool;
	UpdateToolButtons();
	if(current_tool == TOOL_REVIEW)
		OnEnterReviewMode();
	Refresh();
}

void AnnotateView::UpdateToolButtons() {
	btn_select.Set(current_tool == TOOL_SELECT);
	btn_bbox.Set(current_tool == TOOL_BBOX);
	btn_poly.Set(current_tool == TOOL_POLYGON);
	btn_brush.Set(current_tool == TOOL_BRUSH);
	btn_eraser.Set(current_tool == TOOL_ERASER);
	btn_wand.Set(current_tool == TOOL_MAGICWAND);
	btn_keypoint.Set(current_tool == TOOL_KEYPOINT);
	btn_review.Set(current_tool == TOOL_REVIEW);
}

void AnnotateView::SetToolByName(const String& name) {
	if(name == "select")        SetCurrentTool(TOOL_SELECT);
	else if(name == "bbox")     SetCurrentTool(TOOL_BBOX);
	else if(name == "polygon")  SetCurrentTool(TOOL_POLYGON);
	else if(name == "brush")    SetCurrentTool(TOOL_BRUSH);
	else if(name == "eraser")   SetCurrentTool(TOOL_ERASER);
	else if(name == "wand")     SetCurrentTool(TOOL_MAGICWAND);
	else if(name == "keypoint") SetCurrentTool(TOOL_KEYPOINT);
	else if(name == "review")   SetCurrentTool(TOOL_REVIEW);
}

void AnnotateView::SelectById(int id) {
	selected_ids.Clear();
	selected_kp_id = -1;
	if(id < 0) {
		selected_id = -1;
		SyncListSelectionFromCurrent();
		Refresh();
		return;
	}
	if(!entry) return;
	for(const auto& obj : entry->annotations)
		if(obj.id == id) { selected_id = id; selected_ids.Add(id); SyncListSelectionFromCurrent(); Refresh(); return; }
	for(const auto& obj : entry->suggestions)
		if(obj.id == id) { selected_id = id; selected_ids.Add(id); SyncListSelectionFromCurrent(); Refresh(); return; }
	selected_id = -1;
	Refresh();
}

void AnnotateView::SelectAllObjects() {
	selected_ids.Clear();
	selected_kp_id = -1;
	if(!entry) {
		selected_id = -1;
		SyncListSelectionFromCurrent();
		Refresh();
		return;
	}
	for(int i = 0; i < entry->annotations.GetCount(); i++)
		selected_ids.Add(entry->annotations[i].id);
	selected_id = selected_ids.IsEmpty() ? -1 : selected_ids.Top();
	SyncListSelectionFromCurrent();
	Refresh();
}

void AnnotateView::RefreshAfterBulkEdit(ImageEntry& modified_entry) {
	if(entry == &modified_entry) {
		RefreshObjectTree();
		Refresh();
	}
}

bool AnnotateView::Key(dword key, int count) {
	if(key == K_CTRL_A || ((key == 'A' || key == 'a') && GetCtrl())) { SelectAllObjects(); return true; }
	if(key == 'V' || key == 'v') { SetCurrentTool(TOOL_SELECT); return true; }
	if(key == 'B' || key == 'b') { SetCurrentTool(TOOL_BBOX); return true; }
	if(key == 'P' || key == 'p') { SetCurrentTool(TOOL_POLYGON); return true; }
	if(key == 'W' || key == 'w') { SetCurrentTool(TOOL_MAGICWAND); return true; }
	if(key == 'R' || key == 'r') { SetCurrentTool(TOOL_BRUSH); return true; }
	if(key == 'E' || key == 'e') { SetCurrentTool(TOOL_ERASER); return true; }
	if(key == 'K' || key == 'k') { SetCurrentTool(TOOL_KEYPOINT); return true; }
	if(key == '1') { SetCurrentTool(TOOL_SELECT); return true; }
	if(key == '2') { SetCurrentTool(TOOL_BBOX); return true; }
	if(key == '3') { SetCurrentTool(TOOL_POLYGON); return true; }
	if(key == '4') { SetCurrentTool(TOOL_BRUSH); return true; }
	if(key == '5') { SetCurrentTool(TOOL_ERASER); return true; }
	if(key == '6') { SetCurrentTool(TOOL_MAGICWAND); return true; }
	if(key == '7') { SetCurrentTool(TOOL_KEYPOINT); return true; }
	if((key == 'A' || key == 'a') && !GetCtrl()) { OnAcceptSuggestion(); return true; }
	if(key == 'X' || key == 'x') { OnRejectSuggestion(); return true; }
	if(key == K_TAB) {
		if(!entry || entry->annotations.IsEmpty()) return true;
		int idx = -1;
		for(int i = 0; i < entry->annotations.GetCount(); i++) if(entry->annotations[i].id == selected_id) { idx = i; break; }
		int next = (count < 0) ? (idx - 1 + entry->annotations.GetCount()) % entry->annotations.GetCount() : (idx + 1) % entry->annotations.GetCount();
		if(GetShift()) next = (idx - 1 + entry->annotations.GetCount()) % entry->annotations.GetCount();
		else next = (idx + 1) % entry->annotations.GetCount();
		selected_id = entry->annotations[next].id;
		selected_ids.Clear();
		selected_ids.Add(selected_id);
		if(current_tool == TOOL_REVIEW) CenterOnObject(entry->annotations[next]);
		RefreshObjectTree(); Refresh(); return true;
	}
	if(key == 'N' || key == 'n') { JumpToNextIssue(); return true; }
	if(key == 'M' || key == 'm') { if(entry) { entry->reviewed = !entry->reviewed; Refresh(); WhenDirty(); } return true; }
	if(key == 'A' || key == 'a') { PrevImage(); return true; }
	if(key == 'D' || key == 'd') { NextImage(); return true; }
	if(GetAlt() && key >= '1' && key <= '9') { SelectCategory(key - '1'); return true; }
	if(key == K_DELETE) { DeleteSelectedObjects(); return true; }
	if(key == K_CTRL_D) { DuplicateObject(); return true; }
	if(key == K_CTRL_C) { CopyAnnotations(); return true; }
	if(key == K_CTRL_V) { PasteAnnotations(); return true; }
	if(key == 'L' || key == 'l') { CycleVisibility(); return true; }
	if(key == K_ENTER) {
		if(current_poly.GetCount() > 0) {
			ClosePolygon();
		} else if(GetShift()) {
			JumpToBestImage();
		} else if(chk_auto_advance.Get() && entry && !entry->annotations.IsEmpty()) {
			NextImage();
		}
		return true;
	}
	if(key == K_ESCAPE) { CancelActiveInteractions(); return true; }
	return Ctrl::Key(key, count);
}

END_UPP_NAMESPACE
