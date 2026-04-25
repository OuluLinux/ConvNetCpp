#include "AnnotateView.h"

NAMESPACE_UPP

void AnnotateView::DeleteObjectById(int id) {
	if(!entry) return;
	for(int i = 0; i < entry->annotations.GetCount(); i++) {
		if(entry->annotations[i].id == id) {
			if(cmdmgr) {
				cmdmgr->Execute(new DeleteObjectCommand(*entry, i));
				RemoveSelectionId(id);
				RefreshAfterCommand();
			} else {
				entry->annotations.Remove(i);
				entry->has_annotations = !entry->annotations.IsEmpty();
				RemoveSelectionId(id);
				RefreshObjectTree();
				Refresh();
				WhenDirty();
			}
			return;
		}
	}
}

void AnnotateView::AddAnnotationObject(AnnotationObject&& obj) {
	if(!entry) return;
	entry->annotations.Add(pick(obj));
	entry->has_annotations = true;
	selected_id = entry->annotations.Top().id;
	selected_ids.Clear();
	selected_ids.Add(selected_id);
	RefreshObjectTree();
	Refresh();
	WhenCommandExecuted();
}

void AnnotateView::SetMluiScript(const MluiScript& script) {
	mlui_script_ = script;
	RefreshMetadataSchemaFromMlui();
	RefreshImageMetadataUI();
}

void AnnotateView::ClearMluiScript() {
	mlui_script_ = MluiScript();
	RefreshMetadataSchemaFromMlui();
	RefreshImageMetadataUI();
}

void AnnotateView::RefreshImageMetadataUI() {
	metadata_syncing_ = true;
	meta_schema_table.Clear();
	if(!entry) {
		if(meta_fields_.IsEmpty())
			lbl_meta_schema_info.SetLabel("Schema: no image metadata fields in current .mlui");
		else
			lbl_meta_schema_info.SetLabel(Format("Schema: %d field(s) from .mlui", meta_fields_.GetCount()));
		metadata_wizard_mode_ = false;
		metadata_wizard_index_ = -1;
		btn_meta_wizard.SetData(false);
		metadata_syncing_ = false;
		UpdateMetadataModeUI();
		return;
	}

	entry->EnsureMetadataDefaults();
	if(meta_fields_.IsEmpty()) {
		lbl_meta_schema_info.SetLabel("Schema: no image metadata fields in current .mlui");
	} else {
		lbl_meta_schema_info.SetLabel(Format("Schema: %d field(s) from .mlui", meta_fields_.GetCount()));
	}
	for(int i = 0; i < meta_fields_.GetCount(); i++) {
		const ImageMetaField& f = meta_fields_[i];
		String val = f.default_value;
		int q = entry->image_metadata.Find(f.key);
		if(q >= 0)
			val = entry->image_metadata[q];
		else if(!f.default_value.IsEmpty())
			entry->image_metadata.GetAdd(f.key) = f.default_value;
		int row = meta_schema_table.GetCount();
		meta_schema_table.Add(MakeMetaFieldAttrText(f), val);
		if(f.type == "bool") {
			bool def_b = ImageEntry::ParseBoolMeta(f.default_value, false);
			bool b = ImageEntry::ParseBoolMeta(val, def_b);
			String bval = b ? "true" : "false";
			DropList& dl = meta_schema_table.CreateCtrl<DropList>(row, 1, false);
			dl.Add("true", MakeBoolAttrText(true));
			dl.Add("false", MakeBoolAttrText(false));
			dl.NotNull();
			dl.SetData(bval);
			dl <<= THISBACK(CommitImageMetadataFromUI);
			meta_schema_table.Set(row, 1, MakeBoolAttrText(b));
		}
		else if(f.type == "int") {
			int iv = ImageEntry::ParseIntMeta(val, ImageEntry::ParseIntMeta(f.default_value, 0));
			EditIntSpin& ed = meta_schema_table.CreateCtrl<EditIntSpin>(row, 1, false);
			ed.SetInc(1);
			ed.SetData(iv);
			ed <<= THISBACK(CommitImageMetadataFromUI);
			meta_schema_table.Set(row, 1, iv);
		}
		else if(f.type == "double") {
			double dv = ImageEntry::ParseDoubleMeta(val, ImageEntry::ParseDoubleMeta(f.default_value, 0.0));
			EditDoubleSpin& ed = meta_schema_table.CreateCtrl<EditDoubleSpin>(row, 1, false);
			ed.SetInc(0.1);
			ed.SetData(dv);
			ed <<= THISBACK(CommitImageMetadataFromUI);
			meta_schema_table.Set(row, 1, dv);
		}
		else {
			EditString& ed = meta_schema_table.CreateCtrl<EditString>(row, 1, false);
			ed.SetData(val);
			ed <<= THISBACK(CommitImageMetadataFromUI);
			meta_schema_table.Set(row, 1, val);
		}
	}
	metadata_syncing_ = false;
	if(meta_fields_.IsEmpty() && metadata_wizard_mode_) {
		metadata_wizard_mode_ = false;
		metadata_wizard_index_ = -1;
		btn_meta_wizard.SetData(false);
	}
	UpdateMetadataModeUI();
	if(metadata_wizard_mode_) {
		if(metadata_wizard_index_ < 0 || metadata_wizard_index_ >= meta_fields_.GetCount())
			metadata_wizard_index_ = 0;
		ShowMetadataWizardField();
	}
}

void AnnotateView::CommitImageMetadataFromUI() {
	if(metadata_syncing_ || !entry)
		return;
	entry->EnsureMetadataDefaults();
	int field_count = meta_fields_.GetCount();
	if(field_count > meta_schema_table.GetCount()) field_count = meta_schema_table.GetCount();
	for(int i = 0; i < field_count; i++) {
		const ImageMetaField& f = meta_fields_[i];
		Value raw = meta_schema_table.Get(i, 1);
		if(Ctrl* c = meta_schema_table.GetCtrl(i, 1))
			raw = c->GetData();
		String val = raw.ToString();
		if(f.type == "bool") {
			bool def_b = ImageEntry::ParseBoolMeta(f.default_value, false);
			bool bv = ImageEntry::ParseBoolMeta(val, def_b);
			val = bv ? "true" : "false";
			meta_schema_table.Set(i, 1, MakeBoolAttrText(bv));
		}
		else if(f.type == "int") {
			int iv = ImageEntry::ParseIntMeta(val, ImageEntry::ParseIntMeta(f.default_value, 0));
			val = AsString(iv);
			meta_schema_table.Set(i, 1, iv);
		}
		else if(f.type == "double") {
			double dv = ImageEntry::ParseDoubleMeta(val, ImageEntry::ParseDoubleMeta(f.default_value, 0.0));
			val = AsString(dv);
			meta_schema_table.Set(i, 1, dv);
		}
		else {
			meta_schema_table.Set(i, 1, val);
		}
		int q = entry->image_metadata.Find(f.key);
		if(q >= 0) entry->image_metadata[q] = val;
		else entry->image_metadata.Add(f.key, val);
	}
	entry->SyncImageMetadataToLegacy();
	WhenDirty();
}

void AnnotateView::OnEnterReviewMode() {
	if(current_tool == TOOL_REVIEW && entry && !entry->annotations.IsEmpty()) {
		JumpToNextIssue();
	}
}

void AnnotateView::JumpToNextIssue() {
	if(!entry) return;
	int start_idx = -1;
	for(int i = 0; i < entry->annotations.GetCount(); i++) {
		if(entry->annotations[i].id == selected_id) {
			start_idx = i;
			break;
		}
	}
	for(int i = 1; i <= entry->annotations.GetCount(); i++) {
		int idx = (start_idx + i) % entry->annotations.GetCount();
		if(HasIssue(entry->annotations[idx])) {
			selected_id = entry->annotations[idx].id;
			CenterOnObject(entry->annotations[idx]);
			RefreshObjectTree();
			Refresh();
			return;
		}
	}
	if(selected_id == -1) {
		selected_id = entry->annotations[0].id;
		CenterOnObject(entry->annotations[0]);
	} else {
		for(int i = 0; i < entry->annotations.GetCount(); i++) {
			if(entry->annotations[i].id == selected_id) {
				int next = (i + 1) % entry->annotations.GetCount();
				selected_id = entry->annotations[next].id;
				CenterOnObject(entry->annotations[next]);
				break;
			}
		}
	}
	RefreshObjectTree();
	Refresh();
}

void AnnotateView::OnToggleVisibility() {
	if(selected_id == -1 || !entry) return;
	for(int i = 0; i < entry->annotations.GetCount(); i++) {
		if(entry->annotations[i].id == selected_id) {
			entry->annotations[i].visible = !entry->annotations[i].visible;
			RefreshObjectTree();
			Refresh();
			WhenDirty();
			break;
		}
	}
}

void AnnotateView::OnGeneralSettings() {
	if(selected_kp_id != -1) OnKeypointSettings();
	else if(selected_id != -1) OnObjectSettings();
}

void AnnotateView::ClearSelectionState() {
	selected_id = -1;
	selected_kp_id = -1;
	selected_ids.Clear();
}

void AnnotateView::RemoveSelectionId(int id) {
	int fi = selected_ids.Find(id);
	if(fi >= 0) selected_ids.Remove(fi);
	if(selected_id == id)
		selected_id = selected_ids.IsEmpty() ? -1 : selected_ids[selected_ids.GetCount() - 1];
	if(selected_ids.IsEmpty())
		selected_kp_id = -1;
}

bool AnnotateView::DeleteSelectedObjects(bool ask_confirm) {
	if(!entry) return false;
	Vector<int> ids;
	auto HasAnnotation = [&](int id) {
		for(const auto& obj : entry->annotations)
			if(obj.id == id) return true;
		return false;
	};
	auto AddUnique = [&](int id) {
		for(int i = 0; i < ids.GetCount(); i++)
			if(ids[i] == id) return;
		ids.Add(id);
	};
	for(int i = 0; i < selected_ids.GetCount(); i++) {
		int sid = selected_ids[i];
		if(HasAnnotation(sid)) AddUnique(sid);
	}
	if(ids.IsEmpty() && selected_id != -1 && HasAnnotation(selected_id))
		ids.Add(selected_id);
	if(ids.IsEmpty()) return false;

	if(ids.GetCount() == 1) {
		selected_id = ids[0];
		OnObjectDelete();
		return true;
	}

	if(ask_confirm && !PromptOKCancel(Format("Delete %d selected objects?", ids.GetCount())))
		return true;

	Vector<int> idxs;
	for(int i = 0; i < entry->annotations.GetCount(); i++) {
		for(int j = 0; j < ids.GetCount(); j++) {
			if(entry->annotations[i].id == ids[j]) {
				idxs.Add(i);
				break;
			}
		}
	}
	if(idxs.IsEmpty()) return false;
	Sort(idxs);

	if(cmdmgr) {
		if(idxs.GetCount() == entry->annotations.GetCount()) {
			cmdmgr->Execute(new ClearAnnotationsCommand(*entry));
		} else {
			for(int i = idxs.GetCount() - 1; i >= 0; i--)
				cmdmgr->Execute(new DeleteObjectCommand(*entry, idxs[i]));
		}
		ClearSelectionState();
		RefreshAfterCommand();
	} else {
		for(int i = idxs.GetCount() - 1; i >= 0; i--)
			entry->annotations.Remove(idxs[i]);
		entry->has_annotations = !entry->annotations.IsEmpty();
		ClearSelectionState();
		RefreshObjectTree();
		Refresh();
		WhenDirty();
	}
	return true;
}

void AnnotateView::OnGeneralDelete() {
	if(selected_kp_id != -1) OnKeypointDelete();
	else if(selected_ids.GetCount() > 1) DeleteSelectedObjects();
	else if(selected_id != -1) OnObjectDelete();
}

void AnnotateView::OnKeypointSettings() {
	if(selected_kp_id == -1 || !entry) return;
	for(auto& obj : entry->annotations) {
		for(auto& kp : obj.keypoints) {
			if(kp.id == selected_kp_id) {
				KeypointSettingsDialog dlg(kp);
				if(dlg.Run() == IDOK) {
					RefreshObjectTree();
					Refresh();
					WhenDirty();
				}
				return;
			}
		}
	}
}

void AnnotateView::OnKeypointDelete() {
	if(selected_kp_id == -1 || !entry) return;
	if(PromptOKCancel("Delete selected keypoint?")) {
		for(auto& obj : entry->annotations) {
			for(int i = 0; i < obj.keypoints.GetCount(); i++) {
				if(obj.keypoints[i].id == selected_kp_id) {
					if(cmdmgr) {
						cmdmgr->Execute(new DeleteKeypointCommand(*entry, obj.id, obj.keypoints[i]));
						selected_kp_id = -1;
						RefreshAfterCommand();
					} else {
						obj.keypoints.Remove(i);
						obj.UpdateBBox();
						selected_kp_id = -1;
						RefreshObjectTree();
						Refresh();
						WhenDirty();
					}
					return;
				}
			}
		}
	}
}

void AnnotateView::OnObjectSettings() {
	if(selected_id == -1 || !entry || !categories) return;
	for(int i = 0; i < entry->annotations.GetCount(); i++) {
		if(entry->annotations[i].id == selected_id) {
			ObjectSettingsDialog dlg(entry->annotations[i], *categories);
			if(dlg.Run() == IDOK) {
				RefreshObjectTree();
				Refresh();
				WhenDirty();
			}
			break;
		}
	}
}

int AnnotateView::GetNextObjectId() const {
	int max_id = 0;
	if(entry) {
		for(const auto& o : entry->annotations)    max_id = max(max_id, o.id);
		for(const auto& o : entry->suggestions)    max_id = max(max_id, o.id);
	}
	return max_id + 1;
}

void AnnotateView::OnSetGeometry() {
	if(!entry) return;
	SetGeometryDialog dlg;
	if(img) dlg.SetImageSize(img.GetWidth(), img.GetHeight());

	AnnotationObject* target = nullptr;
	bool is_new = false;

	if(selected_id != -1) {
		for(auto& obj : entry->annotations)
			if(obj.id == selected_id) { target = &obj; break; }
	}
	if(!target) {
		entry->annotations.Add();
		target = &entry->annotations.Top();
		target->id = GetNextObjectId();
		if(active_category_id != -1) target->category_id = active_category_id;
		is_new = true;
	} else {
		if(!target->polygons.IsEmpty() && !target->polygons[0].IsEmpty())
			dlg.SetFromPolygon(target->polygons[0]);
		else
			dlg.SetFromBBox(target->bbox);
	}

	if(dlg.Run() != IDOK) {
		if(is_new) entry->annotations.Remove(entry->annotations.GetCount() - 1);
		return;
	}

	Vector<Pointf> poly = dlg.GetPolygon();
	if(poly.IsEmpty()) {
		if(is_new) entry->annotations.Remove(entry->annotations.GetCount() - 1);
		return;
	}

	target->polygons.Clear();
	target->polygons.Add(pick(poly));
	target->UpdateBBox();
	selected_id = target->id;
	entry->has_annotations = true;
	RefreshObjectTree();
	Refresh();
	WhenDirty();
}

int AnnotateView::ApplyMluiScript(const MluiScript& script) {
	if(!entry || !img) return 0;
	int iw = img.GetWidth(), ih = img.GetHeight();
	int added = 0;

	for(const auto& slot : script.slots) {
		bool found = false;
		for(const auto& obj : entry->annotations) {
			if(obj.metadata.Get(MluiSlotIdKey(), "") == slot.slot_id)
				{ found = true; break; }
		}
		if(found && !slot.allow_multiple) continue;

		AnnotationObject obj;
		obj.id          = GetNextObjectId();
		obj.name        = slot.label;
		obj.category_id = -1;
		if(categories) {
			for(const auto& c : *categories)
				if(c.name == slot.category) { obj.category_id = c.id; break; }
		}
		obj.metadata.Add(MluiSlotIdKey(), slot.slot_id);

		if(slot.bbox_hint.Width() > 0 && slot.bbox_hint.Height() > 0) {
			double x0 = slot.bbox_hint.left   * iw;
			double y0 = slot.bbox_hint.top    * ih;
			double x1 = slot.bbox_hint.right  * iw;
			double y1 = slot.bbox_hint.bottom * ih;
			Vector<Pointf> pts;
			pts.Add(Pointf(x0, y0)); pts.Add(Pointf(x1, y0));
			pts.Add(Pointf(x1, y1)); pts.Add(Pointf(x0, y1));
			obj.polygons.Add(pick(pts));
			obj.UpdateBBox();
		}
		entry->annotations.Add(pick(obj));
		entry->has_annotations = true;
		added++;
	}

	if(added > 0) {
		RefreshObjectTree();
		Refresh();
		WhenDirty();
	}
	return added;
}

int AnnotateView::UpdateScriptHintsFromImage(MluiScript& script) {
	if(!entry || !img) return 0;
	int iw = img.GetWidth(), ih = img.GetHeight();
	if(iw <= 0 || ih <= 0) return 0;
	int updated = 0;
	for(auto& slot : script.slots) {
		for(const auto& obj : entry->annotations) {
			bool match = (obj.metadata.Get(MluiSlotIdKey(), "") == slot.slot_id);
			if(!match) continue;
			if(obj.bbox.Width() <= 0 || obj.bbox.Height() <= 0) continue;
			slot.bbox_hint = Rectf(
				obj.bbox.left   / iw,
				obj.bbox.top    / ih,
				obj.bbox.right  / iw,
				obj.bbox.bottom / ih);
			updated++;
			break;
		}
	}
	if(!entry->file_path.IsEmpty()) {
		script.reference_image.file_path = entry->file_path;
		script.reference_image.width  = iw;
		script.reference_image.height = ih;
	}
	return updated;
}

void AnnotateView::PopulateScriptEditorReference(MluiScriptEditor& editor) {
	if(!entry || !img) return;
	Vector<MluiScriptEditor::RefAnnotEntry> entries;
	for(const auto& obj : entry->annotations) {
		if(obj.bbox.Width() <= 0 || obj.bbox.Height() <= 0) continue;
		MluiScriptEditor::RefAnnotEntry e;
		e.slot_id = obj.metadata.Get(MluiSlotIdKey(), "");
		if(e.slot_id.IsEmpty()) e.slot_id = obj.name;
		e.bbox = obj.bbox;
		entries.Add(e);
	}
	editor.SetReferenceAnnotations(entries,
	                               img.GetWidth(), img.GetHeight(),
	                               entry->file_path);
	editor.SetCurrentImageSize(img.GetWidth(), img.GetHeight());
}

void AnnotateView::OnObjectDelete() {
	if(selected_id == -1 || !entry) return;
	if(PromptOKCancel("Delete selected object?")) {
		for(int i = 0; i < entry->annotations.GetCount(); i++) {
			if(entry->annotations[i].id == selected_id) {
				if(cmdmgr) {
					cmdmgr->Execute(new DeleteObjectCommand(*entry, i));
					entry->has_annotations = !entry->annotations.IsEmpty();
					RemoveSelectionId(selected_id);
					RefreshAfterCommand();
				} else {
					entry->annotations.Remove(i);
					entry->has_annotations = !entry->annotations.IsEmpty();
					RemoveSelectionId(selected_id);
					RefreshObjectTree();
					Refresh();
					WhenDirty();
				}
				break;
			}
		}
	}
}

void AnnotateView::OnCopyAnnotations() {
	if(!dataset || current_img_idx == -1 || !categories) return;
	CopyAnnotationsDialog dlg(*dataset, current_img_idx, *categories);
	if(dlg.Run() == IDOK) {
		int target_idx = dlg.GetTargetImageIndex();
		if(target_idx < 0 || target_idx >= dataset->images.GetCount()) return;
		Vector<AnnotationObject> to_copy;
		Vector<int> filter_ids = dlg.GetSelectedCategoryIds();
		bool use_filter = dlg.IsFilterEnabled();
		for(const auto& obj : entry->annotations) {
			bool matched = !use_filter;
			if(use_filter) for(int id : filter_ids) if(id == obj.category_id) { matched = true; break; }
			if(matched) {
				AnnotationObject copy(obj);
				static int next_id = 2000; copy.id = next_id++;
				to_copy.Add(pick(copy));
			}
		}
		if(to_copy.IsEmpty()) { PromptOK("No annotations to copy."); return; }
		if(cmdmgr) {
			cmdmgr->Execute(new CopyAnnotationsCommand(dataset->images[target_idx], to_copy));
			RefreshAfterCommand();
		} else {
			for(auto& o : to_copy) dataset->images[target_idx].annotations.Add(pick(o));
			dataset->images[target_idx].has_annotations = true;
			Refresh();
		}
	}
}

Vector<String> AnnotateView::GetUnfilledSlots(const MluiScript& script) const {
	Vector<String> out;
	for(const auto& slot : script.slots) {
		bool found = false;
		if(entry) {
			for(const auto& obj : entry->annotations) {
				if(obj.metadata.Get(MluiSlotIdKey(), "") == slot.slot_id)
					{ found = true; break; }
			}
		}
		if(!found) out.Add(slot.slot_id);
	}
	return out;
}

String AnnotateView::ResolveImagePath(const String& path) const {
	if(WhenResolveImagePath) {
		String p = WhenResolveImagePath(path);
		if(!p.IsEmpty()) return p;
	}
	return path;
}

END_UPP_NAMESPACE
