#include "AnnotateView.h"

NAMESPACE_UPP

AnnotateView::AnnotateView() {
	Add(toolbar.TopPos(0, 30).HSizePos());
	toolbar.SetFrame(FieldFrame());

	toolbar.Add(btn_back.SetLabel("< Back").LeftPos(5, 60).TopPos(5, 20));
	toolbar.Add(btn_prev.SetLabel("< Prev").LeftPos(70, 60).TopPos(5, 20));
	toolbar.Add(btn_next.SetLabel("Next >").LeftPos(135, 60).TopPos(5, 20));

	toolbar.Add(btn_select.SetLabel("Select").LeftPos(205, 60).TopPos(5, 20));
	toolbar.Add(btn_bbox.SetLabel("BBox").LeftPos(270, 60).TopPos(5, 20));
	toolbar.Add(btn_poly.SetLabel("Polygon").LeftPos(335, 60).TopPos(5, 20));
	toolbar.Add(btn_brush.SetLabel("Brush").LeftPos(400, 60).TopPos(5, 20));
	toolbar.Add(btn_eraser.SetLabel("Eraser").LeftPos(465, 60).TopPos(5, 20));
	toolbar.Add(btn_wand.SetLabel("Wand").LeftPos(530, 50).TopPos(5, 20));
	toolbar.Add(btn_keypoint.SetLabel("KP").LeftPos(585, 40).TopPos(5, 20));
	toolbar.Add(btn_review.SetLabel("Review").LeftPos(630, 60).TopPos(5, 20));
	toolbar.Add(btn_close_poly.SetLabel("Close").LeftPos(695, 60).TopPos(5, 20));
	toolbar.Add(btn_copy.SetLabel("Copy").LeftPos(760, 60).TopPos(5, 20));

	toolbar.Add(btn_center.SetLabel("Center").LeftPos(825, 60).TopPos(5, 20));

	toolbar.Add(lbl_active_tool.LeftPos(890, 80).TopPos(5, 20));
	lbl_active_tool.SetFont(StdFont().Bold());

	toolbar.Add(btn_undo.SetLabel("Undo").RightPos(75, 60).TopPos(5, 20));
	toolbar.Add(btn_redo.SetLabel("Redo").RightPos(10, 60).TopPos(5, 20));

	btn_back << [=] { WhenBack(); };
	btn_prev << THISBACK(PrevImage);
	btn_next << THISBACK(NextImage);

	btn_select << [=] { SetCurrentTool(TOOL_SELECT); };
	btn_bbox << [=] { SetCurrentTool(TOOL_BBOX); };
	btn_poly << [=] { SetCurrentTool(TOOL_POLYGON); };
	btn_brush << [=] { SetCurrentTool(TOOL_BRUSH); };
	btn_eraser << [=] { SetCurrentTool(TOOL_ERASER); };
	btn_wand << [=] { SetCurrentTool(TOOL_MAGICWAND); };
	btn_keypoint << [=] { SetCurrentTool(TOOL_KEYPOINT); };
	btn_review << [=] { SetCurrentTool(TOOL_REVIEW); };
	btn_close_poly << THISBACK(ClosePolygon);
	btn_copy << THISBACK(OnCopyAnnotations);
	btn_center << [=] { CenterImage(); };

	btn_undo << THISBACK(OnUndo);
	btn_redo << THISBACK(OnRedo);
	UpdateToolButtons();

	categories_panel.SetFrame(FieldFrame());
	categories_panel.Add(lbl_cat_title.SetLabel("Categories").HSizePos(5, 5).TopPos(5, 20));
	lbl_cat_title.SetFont(StdFont().Bold());
	categories_panel.Add(cat_list.HSizePos(5, 5).VSizePos(30, 35));
	cat_list.AddColumn("Name");
	cat_list.WhenSel = THISBACK(OnCategorySel);

	categories_panel.Add(btn_cat_settings.SetLabel("Settings").RightPos(5, 90).BottomPos(5, 25));
	btn_cat_settings << THISBACK(OnCategorySettings);

	settings_panel.SetFrame(FieldFrame());
	settings_panel.Add(lbl_wand_settings.SetLabel("Wand Settings").HSizePos(5, 5).TopPos(8, 20));
	lbl_wand_settings.SetFont(StdFont().Bold());
	settings_panel.Add(lbl_threshold.SetLabel("Threshold:").LeftPos(5, 100).TopPos(34, 20));
	settings_panel.Add(edit_threshold.RightPos(5, 60).TopPos(34, 20));
	edit_threshold.SetData(25);

	settings_panel.Add(lbl_view_settings.SetLabel("View Settings").HSizePos(5, 5).TopPos(66, 20));
	lbl_view_settings.SetFont(StdFont().Bold());
	settings_panel.Add(chk_show_hover.SetLabel("Show Hover Info").HSizePos(5, 5).TopPos(92, 20));
	chk_show_hover.SetData(true);
	chk_show_hover << [=] { show_hover_info = chk_show_hover.Get(); Refresh(); };

	settings_panel.Add(lbl_flow_settings.SetLabel("Flow Settings").HSizePos(5, 5).TopPos(124, 20));
	lbl_flow_settings.SetFont(StdFont().Bold());
	settings_panel.Add(chk_auto_create.SetLabel("Auto-create Object").HSizePos(5, 5).TopPos(150, 20));
	chk_auto_create.SetData(true);
	settings_panel.Add(chk_auto_advance.SetLabel("Auto-advance Image").HSizePos(5, 5).TopPos(176, 20));
	chk_auto_advance.SetData(true);

	metadata_panel.SetFrame(FieldFrame());
	metadata_panel.Add(lbl_meta_title.SetLabel("Image Metadata").HSizePos(5, 5).TopPos(8, 20));
	lbl_meta_title.SetFont(StdFont().Bold());
	metadata_panel.Add(lbl_currency.SetLabel("Currency:").LeftPos(5, 100).TopPos(34, 20));
	metadata_panel.Add(edit_currency.RightPos(5, 120).TopPos(34, 20));
	metadata_panel.Add(lbl_dealer.SetLabel("Dealer:").LeftPos(5, 100).TopPos(58, 20));
	metadata_panel.Add(edit_dealer.RightPos(5, 120).TopPos(58, 20));
	edit_dealer.MinMax(-1, 20);
	metadata_panel.Add(lbl_pot_total.SetLabel("Pot total:").LeftPos(5, 100).TopPos(82, 20));
	metadata_panel.Add(edit_pot_total.RightPos(5, 120).TopPos(82, 20));
	metadata_panel.Add(lbl_round_pot.SetLabel("Round pot:").LeftPos(5, 100).TopPos(106, 20));
	metadata_panel.Add(edit_round_pot.RightPos(5, 120).TopPos(106, 20));
	metadata_panel.Add(lbl_side_pot.SetLabel("Side pot:").LeftPos(5, 100).TopPos(130, 20));
	metadata_panel.Add(edit_side_pot.RightPos(5, 120).TopPos(130, 20));
	metadata_panel.Add(lbl_hero_cards.SetLabel("Hero cards:").LeftPos(5, 100).TopPos(154, 20));
	metadata_panel.Add(edit_hero_cards.RightPos(5, 220).TopPos(154, 20));
	metadata_panel.Add(lbl_board_cards.SetLabel("Board cards:").LeftPos(5, 100).TopPos(178, 20));
	metadata_panel.Add(edit_board_cards.RightPos(5, 220).TopPos(178, 20));
	metadata_panel.Add(lbl_players_title.SetLabel("Players (T=In table, G=In game)").HSizePos(5, 5).TopPos(204, 18));
	lbl_players_title.SetFont(StdFont().Bold());
	for(int i = 0; i < kMaxPlayerControls; i++) {
		int col = i / 4;
		int row = i % 4;
		int x = 5 + col * 165;
		int y = 226 + row * 22;
		metadata_panel.Add(lbl_player_row[i].SetLabel(Format("P%d", i + 1)).LeftPos(x, 24).TopPos(y, 18));
		metadata_panel.Add(opt_player_in_table[i].SetLabel("T").LeftPos(x + 28, 46).TopPos(y, 18));
		metadata_panel.Add(opt_player_in_game[i].SetLabel("G").LeftPos(x + 78, 46).TopPos(y, 18));
	}

	edit_currency <<= THISBACK(CommitImageMetadataFromUI);
	edit_dealer <<= THISBACK(CommitImageMetadataFromUI);
	edit_pot_total <<= THISBACK(CommitImageMetadataFromUI);
	edit_round_pot <<= THISBACK(CommitImageMetadataFromUI);
	edit_side_pot <<= THISBACK(CommitImageMetadataFromUI);
	edit_hero_cards <<= THISBACK(CommitImageMetadataFromUI);
	edit_board_cards <<= THISBACK(CommitImageMetadataFromUI);
	for(int i = 0; i < kMaxPlayerControls; i++) {
		opt_player_in_table[i] <<= THISBACK(CommitImageMetadataFromUI);
		opt_player_in_game[i] <<= THISBACK(CommitImageMetadataFromUI);
	}
	metadata_panel.Add(btn_meta_wizard.SetLabel("Wizard Mode").RightPos(5, 110).TopPos(32, 20));
	metadata_panel.Add(lbl_meta_schema_info.SetLabel("Schema: (loading .mlui...)").HSizePos(5, 5).TopPos(34, 18));
	metadata_panel.Add(meta_schema_table.HSizePos(5, 5).VSizePos(56, 5));
	meta_schema_table.AddColumn("Field", 170);
	meta_schema_table.AddColumn("Value", 320);
	meta_schema_table.SetLineCy(20);
	meta_schema_table.WhenAcceptEdit = THISBACK(CommitImageMetadataFromUI);
	meta_schema_table.WhenCtrlsAction = THISBACK(CommitImageMetadataFromUI);
	btn_meta_wizard <<= THISBACK(OnToggleMetadataWizardMode);

	metadata_panel.Add(lbl_meta_wizard_field.HSizePos(5, 5).TopPos(62, 20));
	lbl_meta_wizard_field.SetFont(StdFont().Bold());
	metadata_panel.Add(lbl_meta_wizard_hint.HSizePos(5, 5).TopPos(84, 20));
	metadata_panel.Add(edit_meta_wizard_string.HSizePos(5, 5).TopPos(108, 24));
	metadata_panel.Add(edit_meta_wizard_bool.HSizePos(5, 5).TopPos(108, 24));
	metadata_panel.Add(edit_meta_wizard_int.HSizePos(5, 5).TopPos(108, 24));
	metadata_panel.Add(edit_meta_wizard_double.HSizePos(5, 5).TopPos(108, 24));
	edit_meta_wizard_int.SetInc(1);
	edit_meta_wizard_double.SetInc(0.1);
	edit_meta_wizard_string.WhenEnter = THISBACK(OnMetadataWizardEnter);
	edit_meta_wizard_bool.WhenEnter = THISBACK(OnMetadataWizardEnter);
	edit_meta_wizard_int.WhenEnter = THISBACK(OnMetadataWizardEnter);
	edit_meta_wizard_double.WhenEnter = THISBACK(OnMetadataWizardEnter);
	lbl_meta_wizard_field.Hide();
	lbl_meta_wizard_hint.Hide();
	edit_meta_wizard_string.Hide();
	edit_meta_wizard_bool.Hide();
	edit_meta_wizard_int.Hide();
	edit_meta_wizard_double.Hide();

	lbl_currency.Hide(); edit_currency.Hide();
	lbl_dealer.Hide(); edit_dealer.Hide();
	lbl_pot_total.Hide(); edit_pot_total.Hide();
	lbl_round_pot.Hide(); edit_round_pot.Hide();
	lbl_side_pot.Hide(); edit_side_pot.Hide();
	lbl_hero_cards.Hide(); edit_hero_cards.Hide();
	lbl_board_cards.Hide(); edit_board_cards.Hide();
	lbl_players_title.Hide();
	for(int i = 0; i < kMaxPlayerControls; i++) {
		lbl_player_row[i].Hide();
		opt_player_in_table[i].Hide();
		opt_player_in_game[i].Hide();
	}

	objects_panel.SetFrame(FieldFrame());
	objects_panel.Add(lbl_tree_title.SetLabel("Objects").HSizePos(5, 5).TopPos(5, 20));
	lbl_tree_title.SetFont(StdFont().Bold());
	objects_panel.Add(obj_tree.HSizePos(5, 5).VSizePos(30, 70));
	obj_tree.Header();
	obj_tree.HeaderTab(0).SetText("Object");
	obj_tree.AddColumn("Rect", 90);
	obj_tree.AddColumn("Text", 140);
	obj_tree.WhenCursor << THISBACK(OnTreeSel);
	obj_tree.WhenLeftDouble = [=] { OnGeneralSettings(); };

	objects_panel.Add(btn_obj_add.SetLabel("+ Add").LeftPos(5, 80).BottomPos(5, 25));
	objects_panel.Add(btn_obj_delete.SetLabel("Delete").LeftPos(90, 70).BottomPos(5, 25));
	objects_panel.Add(btn_obj_visible.SetLabel("Visible").LeftPos(165, 70).BottomPos(5, 25));
	objects_panel.Add(btn_obj_settings.SetLabel("Settings").LeftPos(5, 90).BottomPos(33, 25));
	objects_panel.Add(btn_obj_geometry.SetLabel("Set Geometry").LeftPos(100, 110).BottomPos(33, 25));

	btn_obj_visible << THISBACK(OnToggleVisibility);
	btn_obj_settings << THISBACK(OnGeneralSettings);
	btn_obj_delete << THISBACK(OnGeneralDelete);
	btn_obj_geometry << THISBACK(OnSetGeometry);
	btn_obj_add << [=] { ClearSelectionState(); OnSetGeometry(); };
	RefreshMetadataSchemaFromMlui();
	UpdateMetadataModeUI();

	BackPaint();
}

void AnnotateView::OnUndo() {
	if(cmdmgr) {
		cmdmgr->Undo();
		RefreshAfterCommand();
		WhenLog("Undo: " + GetLastUndoName());
	}
}

void AnnotateView::OnRedo() {
	if(cmdmgr) {
		cmdmgr->Redo();
		RefreshAfterCommand();
		WhenLog("Redo: " + GetLastRedoName());
	}
}

String AnnotateView::GetLastUndoName() {
	if(cmdmgr && cmdmgr->CanUndo())
		return cmdmgr->GetUndoStack().Top().GetName();
	return "";
}

String AnnotateView::GetLastRedoName() {
	if(cmdmgr && cmdmgr->CanRedo())
		return cmdmgr->GetRedoStack().Top().GetName();
	return "";
}

void AnnotateView::RefreshAfterCommand() {
	RefreshObjectTree();
	Refresh();
	WhenCommandExecuted();
}

void AnnotateView::SetCategories(Vector<Category>& c) {
	categories = &c;
	RefreshCategoryList();
}

void AnnotateView::SetDataset(Dataset& ds, int img_idx) {
	dataset = &ds;
	current_img_idx = img_idx;
	entry = &ds.images[img_idx];
	img = StreamRaster::LoadFileAny(ResolveImagePath(entry->file_path));
	InvalidateScaledImage();

	current_poly.Clear();
	drawing_bbox = false;
	selecting_rect = false;
	brushing = false;
	moving_objects = false;
	move_obj_indices.Clear();
	move_old_polys.Clear();
	move_old_kps.Clear();
	selected_id = -1;
	selected_ids.Clear();
	selected_kp_id = -1;

	bool can_cycle = dataset->images.GetCount() > 1;
	btn_prev.Enable(can_cycle);
	btn_next.Enable(can_cycle);

	CenterImage();
	PostCallback(THISBACK(CenterImage));
	RefreshCategoryList();
	RefreshObjectTree();
	RefreshImageMetadataUI();
	Refresh();
	WhenLog(Format("Opened image %d/%d: %s", current_img_idx + 1, dataset->images.GetCount(), entry->file_name));
}

bool AnnotateView::ParseSeatFromSlotId(const String& sid, int& seat_zero_based, String* suffix) const {
	String s = ToLower(sid);
	if(!s.StartsWith("seat"))
		return false;
	int i = 4;
	String num;
	while(i < s.GetCount() && IsDigit((byte)s[i])) {
		num.Cat(s[i]);
		i++;
	}
	if(num.IsEmpty())
		return false;
	if(i >= s.GetCount() || s[i] != '_')
		return false;
	if(suffix)
		*suffix = s.Mid(i + 1);
	int seat = StrInt(num);
	if(seat <= 0)
		return false;
	seat_zero_based = seat - 1;
	return true;
}

bool AnnotateView::TryParseWizardBool(const String& raw, bool& out) const {
	String s = ToLower(TrimBoth(raw));
	if(s == "1" || s == "t" || s == "true" || s == "yes" || s == "on") {
		out = true;
		return true;
	}
	if(s == "0" || s == "f" || s == "false" || s == "no" || s == "off") {
		out = false;
		return true;
	}
	return false;
}

Color AnnotateView::AdjustColorBrightness(Color c, int delta) const {
	auto clamp = [&](int v) { return minmax(v, 0, 255); };
	return Color(clamp(c.GetR() + delta), clamp(c.GetG() + delta), clamp(c.GetB() + delta));
}

Color AnnotateView::SeatBaseColor(int seat_zero_based) const {
	static const Color palette[] = {
		Color(72, 122, 232),
		Color(66, 176, 201),
		Color(60, 182, 125),
		Color(120, 180, 78),
		Color(188, 152, 60),
		Color(208, 118, 62),
		Color(198, 90, 118),
		Color(150, 96, 200),
	};
	const int n = (int)(sizeof(palette) / sizeof(palette[0]));
	if(seat_zero_based < 0)
		seat_zero_based = 0;
	return palette[seat_zero_based % n];
}

bool AnnotateView::GetSeatFieldColor(const String& key, Color& out) const {
	int seat_i = -1;
	String suffix;
	if(!ParseSeatFromSlotId(ToLower(key), seat_i, &suffix))
		return false;
	Color base = SeatBaseColor(seat_i);
	suffix = ToLower(suffix);
	int delta = 14;
	if(suffix == "in_table")
		delta = 28;
	else if(suffix == "in_game")
		delta = 6;
	else if(suffix == "bet_chip" || suffix == "bet")
		delta = -14;
	out = AdjustColorBrightness(base, delta);
	return true;
}

AttrText AnnotateView::MakeMetaFieldAttrText(const ImageMetaField& f) const {
	AttrText t(f.label + " [" + f.key + "]");
	Color c;
	if(GetSeatFieldColor(f.key, c))
		t.Ink(c).Bold();
	return t;
}

AttrText AnnotateView::MakeBoolAttrText(bool value) const {
	if(value)
		return AttrText("True").Ink(Color(34, 146, 54)).Bold();
	return AttrText("False").Ink(Color(190, 44, 44)).Bold();
}

void AnnotateView::UpdateMetadataModeUI() {
	bool wizard_active = metadata_wizard_mode_ && entry && !meta_fields_.IsEmpty();
	lbl_meta_schema_info.Show(!wizard_active);
	meta_schema_table.Show(!wizard_active);
	lbl_meta_wizard_field.Show(wizard_active);
	lbl_meta_wizard_hint.Show(wizard_active);
	if(!wizard_active) {
		edit_meta_wizard_string.Hide();
		edit_meta_wizard_bool.Hide();
		edit_meta_wizard_int.Hide();
		edit_meta_wizard_double.Hide();
	}
}

void AnnotateView::ShowMetadataWizardEditor(const String& type, const String& value) {
	edit_meta_wizard_string.Hide();
	edit_meta_wizard_bool.Hide();
	edit_meta_wizard_int.Hide();
	edit_meta_wizard_double.Hide();
	if(type == "int") {
		edit_meta_wizard_int.Show();
		edit_meta_wizard_int.SetData(ImageEntry::ParseIntMeta(value, 0));
		edit_meta_wizard_int.SetFocus();
	}
	else if(type == "double") {
		edit_meta_wizard_double.Show();
		edit_meta_wizard_double.SetData(ImageEntry::ParseDoubleMeta(value, 0.0));
		edit_meta_wizard_double.SetFocus();
	}
	else if(type == "bool") {
		bool b = ImageEntry::ParseBoolMeta(value, false);
		edit_meta_wizard_bool.Show();
		edit_meta_wizard_bool.SetData(b ? "t" : "f");
		edit_meta_wizard_bool.SetFocus();
	}
	else {
		edit_meta_wizard_string.Show();
		edit_meta_wizard_string.SetData(value);
		edit_meta_wizard_string.SetFocus();
	}
}

void AnnotateView::ShowMetadataWizardField() {
	if(!metadata_wizard_mode_ || !entry || meta_fields_.IsEmpty()) {
		UpdateMetadataModeUI();
		return;
	}
	if(metadata_wizard_index_ < 0)
		metadata_wizard_index_ = 0;
	if(metadata_wizard_index_ >= meta_fields_.GetCount())
		metadata_wizard_index_ = meta_fields_.GetCount() - 1;
	const ImageMetaField& f = meta_fields_[metadata_wizard_index_];
	String val = f.default_value;
	int q = entry->image_metadata.Find(f.key);
	if(q >= 0)
		val = entry->image_metadata[q];
	lbl_meta_wizard_field.SetLabel(Format("Field %d/%d: %s [%s]",
	                                      metadata_wizard_index_ + 1, meta_fields_.GetCount(),
	                                      f.label, f.key));
	if(f.type == "bool")
		lbl_meta_wizard_hint.SetLabel("Enter boolean as 1, 0, t, or f, then press Enter.");
	else
		lbl_meta_wizard_hint.SetLabel("Press Enter to save this value and jump to the next field.");
	UpdateMetadataModeUI();
	ShowMetadataWizardEditor(f.type, val);
}

bool AnnotateView::CommitMetadataWizardField(bool advance) {
	if(!metadata_wizard_mode_ || !entry || meta_fields_.IsEmpty())
		return false;
	if(metadata_wizard_index_ < 0 || metadata_wizard_index_ >= meta_fields_.GetCount())
		return false;
	const ImageMetaField& f = meta_fields_[metadata_wizard_index_];
	String val;
	if(f.type == "int") {
		int iv = ImageEntry::ParseIntMeta(edit_meta_wizard_int.GetData().ToString(),
		                                  ImageEntry::ParseIntMeta(f.default_value, 0));
		val = AsString(iv);
	}
	else if(f.type == "double") {
		double dv = ImageEntry::ParseDoubleMeta(edit_meta_wizard_double.GetData().ToString(),
		                                        ImageEntry::ParseDoubleMeta(f.default_value, 0.0));
		val = AsString(dv);
	}
	else if(f.type == "bool") {
		bool b = false;
		if(!TryParseWizardBool(edit_meta_wizard_bool.GetData().ToString(), b)) {
			Exclamation("Boolean value must be one of: 1, 0, t, f.");
			edit_meta_wizard_bool.SetFocus();
			return false;
		}
		val = b ? "true" : "false";
	}
	else {
		val = edit_meta_wizard_string.GetData().ToString();
	}

	int q = entry->image_metadata.Find(f.key);
	if(q >= 0)
		entry->image_metadata[q] = val;
	else
		entry->image_metadata.Add(f.key, val);
	if(metadata_wizard_index_ >= 0 && metadata_wizard_index_ < meta_schema_table.GetCount())
		meta_schema_table.Set(metadata_wizard_index_, 1, f.type == "bool" ? (Value)MakeBoolAttrText(val == "true") : (Value)val);
	entry->SyncImageMetadataToLegacy();
	WhenDirty();

	if(!advance)
		return true;
	metadata_wizard_index_++;
	if(metadata_wizard_index_ >= meta_fields_.GetCount()) {
		metadata_wizard_mode_ = false;
		btn_meta_wizard.SetData(false);
		metadata_wizard_index_ = -1;
		RefreshImageMetadataUI();
		UpdateMetadataModeUI();
		return true;
	}
	ShowMetadataWizardField();
	return true;
}

void AnnotateView::OnMetadataWizardEnter() {
	CommitMetadataWizardField(true);
}

void AnnotateView::OnToggleMetadataWizardMode() {
	bool want = (int)btn_meta_wizard.GetData() != 0;
	if(!want) {
		CommitMetadataWizardField(false);
		metadata_wizard_mode_ = false;
		metadata_wizard_index_ = -1;
		UpdateMetadataModeUI();
		return;
	}
	if(!entry || meta_fields_.IsEmpty()) {
		btn_meta_wizard.SetData(false);
		metadata_wizard_mode_ = false;
		metadata_wizard_index_ = -1;
		UpdateMetadataModeUI();
		return;
	}
	metadata_wizard_mode_ = true;
	metadata_wizard_index_ = 0;
	ShowMetadataWizardField();
}

void AnnotateView::RefreshMetadataSchemaFromMlui() {
	meta_fields_.Clear();
	auto CanonicalType = [&](String t) -> String {
		t = ToLower(TrimBoth(t));
		if(t == "bool" || t == "boolean") return "bool";
		if(t == "int" || t == "integer") return "int";
		if(t == "double" || t == "float" || t == "number") return "double";
		if(t == "cards" || t == "cardlist") return "cards";
		return "string";
	};
	auto AddField = [&](String key, String label, String type, String def) {
		key = TrimBoth(key);
		if(key.IsEmpty()) return;
		for(const auto& f : meta_fields_)
			if(f.key == key) return;
		ImageMetaField& f = meta_fields_.Add();
		f.key = key;
		f.label = TrimBoth(label).IsEmpty() ? key : TrimBoth(label);
		f.type = CanonicalType(type);
		f.default_value = def;
	};

	for(const auto& slot : mlui_script_.slots) {
		String key = slot.metadata.Get("image_meta_key", "");
		if(key.IsEmpty())
			key = slot.metadata.Get("img_meta_key", "");
		if(key.IsEmpty())
			continue;
		String label = slot.metadata.Get("image_meta_label", "");
		if(label.IsEmpty())
			label = slot.label;
		String type = slot.metadata.Get("image_meta_type", "string");
		String def = slot.metadata.Get("image_meta_default", "");
		AddField(key, label, type, def);
	}

	if(meta_fields_.IsEmpty()) {
		bool has_dealer = false;
		bool has_total_pot = false;
		bool has_round_pot = false;
		bool has_side_pot = false;
		bool has_card_slots = false;
		int seat_count = 0;
		bool seat_in_table[kMaxPlayerControls];
		bool seat_active[kMaxPlayerControls];
		for(int i = 0; i < kMaxPlayerControls; i++) {
			seat_in_table[i] = false;
			seat_active[i] = false;
		}
		for(const auto& slot : mlui_script_.slots) {
			String sid = ToLower(slot.slot_id);
			if(sid == "dealer_chip") has_dealer = true;
			else if(sid == "total_pot") has_total_pot = true;
			else if(sid == "pot_previous") has_round_pot = true;
			else if(sid == "side_pot") has_side_pot = true;
			else if(sid.Find("_card_") >= 0) has_card_slots = true;

			int seat_i = -1;
			String suffix;
			if(ParseSeatFromSlotId(sid, seat_i, &suffix) && seat_i >= 0 && seat_i < kMaxPlayerControls) {
				seat_count = max(seat_count, seat_i + 1);
				if(suffix == "in_game") seat_active[seat_i] = true;
				if(suffix == "panel" || suffix == "name" || suffix == "balance" || suffix == "bet" || suffix == "in_game")
					seat_in_table[seat_i] = true;
			}
		}
		if(seat_count <= 0)
			seat_count = kMaxPlayerControls;

		AddField("table_currency_unit", "Currency", "string", "BB");
		if(has_dealer) AddField("dealer", "Dealer", "int", "-1");
		if(has_total_pot) AddField("pot_total", "Pot total", "double", "0");
		if(has_round_pot) AddField("round_pot", "Round pot", "double", "0");
		if(has_side_pot) AddField("side_pot", "Side pot", "double", "0");
		if(has_card_slots) AddField("cards", "Cards", "cards", "");
		int seat_limit = seat_count;
		if(seat_limit > kMaxPlayerControls)
			seat_limit = kMaxPlayerControls;
		for(int i = 0; i < seat_limit; i++) {
			if(seat_in_table[i]) AddField(Format("seat%d_in_table", i + 1), Format("Seat %d in table", i + 1), "bool", "true");
			if(seat_active[i]) AddField(Format("seat%d_%s", i + 1, "in_game"), Format("Seat %d in game", i + 1), "bool", "false");
		}
	}
	RefreshImageMetadataUI();
}

bool AnnotateView::HasIssue(const AnnotationObject& obj) {
	if(obj.polygons.IsEmpty())
		return true;
	for(const auto& p : obj.polygons)
		if(p.GetCount() < 3)
			return true;
	if(obj.category_id == -1)
		return true;
	Category* cat = nullptr;
	for(auto& c : *categories)
		if(c.id == obj.category_id)
			cat = &c;
	if(cat && !cat->keypoint_labels.IsEmpty() && obj.keypoints.IsEmpty())
		return true;
	return false;
}

void AnnotateView::CenterOnObject(const AnnotationObject& obj) {
	if(!img)
		return;
	Rectf b = obj.bbox;
	if(b.Width() > 0 && b.Height() > 0) {
		Size sz = GetSize();
		if(sz.cx <= 0)
			sz.cx = 1;
		sz.cy = max(1, sz.cy - 30);
		double ratioX = b.Width() > 0 ? (double)sz.cx / b.Width() : zoom;
		double ratioY = b.Height() > 0 ? (double)sz.cy / b.Height() : zoom;
		double target_ratio = min(ratioX, ratioY);
		if(target_ratio <= 0)
			target_ratio = 1.0;
		zoom = min(max(target_ratio * 0.7, 0.05), 20.0);
		offset.x = (sz.cx / 2.0) - (b.left + b.Width() / 2.0) * zoom;
		offset.y = (sz.cy / 2.0) - (b.top + b.Height() / 2.0) * zoom + 30;
	}
}

void AnnotateView::PanToObject(const AnnotationObject& obj) {
	if(!img)
		return;
	Rectf b = obj.bbox;
	if(b.Width() > 0 && b.Height() > 0) {
		Size sz = GetSize();
		if(sz.cx <= 0)
			sz.cx = 1;
		sz.cy = max(1, sz.cy - 30);
		offset.x = (sz.cx / 2.0) - (b.left + b.Width() / 2.0) * zoom;
		offset.y = (sz.cy / 2.0) - (b.top + b.Height() / 2.0) * zoom + 30;
	}
}

void AnnotateView::PrevImage() {
	if(!dataset)
		return;
	int n = dataset->images.GetCount();
	if(n <= 1)
		return;
	int prev = (current_img_idx - 1 + n) % n;
	WhenOpenImage(prev);
}

void AnnotateView::NextImage() {
	if(!dataset)
		return;
	int n = dataset->images.GetCount();
	if(n <= 1)
		return;
	if(GetShift())
		JumpToBestImage();
	else {
		int next = (current_img_idx + 1) % n;
		WhenOpenImage(next);
	}
}

void AnnotateView::JumpToBestImage() {
	if(!dataset || dataset->images.IsEmpty())
		return;
	int best_idx = -1;
	double best_score = -2.0;
	for(int i = 0; i < dataset->images.GetCount(); i++) {
		if(i == current_img_idx)
			continue;
		if(dataset->images[i].priority_score > best_score) {
			best_score = dataset->images[i].priority_score;
			best_idx = i;
		}
	}
	if(best_idx != -1 && best_score > 0) {
		const auto& next_ie = dataset->images[best_idx];
		String feedback = "Next: ";
		if(next_ie.suggestions.IsEmpty())
			feedback << "Unannotated image";
		else {
			double min_conf = 1.0;
			int cat_id = -1;
			for(const auto& s : next_ie.suggestions)
				if(s.score < min_conf) {
					min_conf = s.score;
					cat_id = s.category_id;
				}
			String cat_name = "Unknown";
			if(categories)
				for(const auto& c : *categories)
					if(c.id == cat_id)
						cat_name = c.name;
			feedback << "High uncertainty (" << cat_name << ", conf ~" << Format("%.2f", min_conf) << ")";
		}
		ShowTemporaryFeedback(feedback);
		WhenOpenImage(best_idx);
	}
	else {
		int n = dataset->images.GetCount();
		if(n > 1) {
			int next = (current_img_idx + 1) % n;
			WhenOpenImage(next);
		}
	}
}

void AnnotateView::ShowTemporaryFeedback(const String& text) {
	temp_overlay_text = text;
	temp_overlay_timeout = 90;
	Refresh();
}

void AnnotateView::RefreshCategoryList() {
	cat_list.Clear();
	if(!categories || !entry)
		return;
	VectorMap<int, int> counts;
	for(const auto& obj : entry->annotations)
		counts.GetAdd(obj.category_id, 0)++;

	for(int i = 0; i < categories->GetCount(); i++) {
		int cid = (*categories)[i].id;
		String name = (*categories)[i].name;
		int count = counts.Get(cid, 0);
		if(count > 0)
			name << " (" << count << ")";
		cat_list.Add(name);
	}
	if(active_category_id != -1) {
		for(int i = 0; i < categories->GetCount(); i++) {
			if((*categories)[i].id == active_category_id) {
				cat_list.SetCursor(i);
				break;
			}
		}
	}
}

String AnnotateView::GetStateText(int state) {
	switch(state) {
		case 0: return "[Not labeled]";
		case 1: return "[Labeled hidden]";
		case 2: return "[Labeled visible]";
		default: return "";
	}
}

void AnnotateView::RegisterNodeMeta(int node, NodeMeta::Type type, int id, int extra) {
	if(node >= 0)
		node_meta.Add(node, NodeMeta{type, id, extra});
}

void AnnotateView::RefreshObjectTree() {
	obj_tree.Clear();
	node_meta.Clear();
	ann_node_by_id.Clear();
	sug_node_by_id.Clear();
	kp_node_by_id.Clear();
	int root = obj_tree.Add(0, Null, Null, "Objects");
	obj_tree.Open(root);
	RegisterNodeMeta(root, NodeMeta::UNKNOWN, -1);
	if(!entry || !categories)
		return;

	VectorMap<int, int> cat_nodes;
	for(int i = 0; i < categories->GetCount(); i++) {
		int node = obj_tree.Add(root, Null, Null, (*categories)[i].name);
		cat_nodes.Add((*categories)[i].id, node);
		obj_tree.Open(node);
		RegisterNodeMeta(node, NodeMeta::CATEGORY, (*categories)[i].id);
	}

	for(int i = 0; i < entry->annotations.GetCount(); i++) {
		AnnotationObject& obj = entry->annotations[i];
		int parent = cat_nodes.Get(obj.category_id, root);
		String label = obj.name;
		if(!obj.visible)
			label << " (Hidden)";
		label << " " << GetStateText(obj.label_visibility_state);
		if(HasIssue(obj))
			label << " [!]";
		int obj_node = obj_tree.Add(parent, Null, obj.id + 1000000, label);
		ann_node_by_id.Add(obj.id, obj_node);
		RegisterNodeMeta(obj_node, NodeMeta::OBJECT, obj.id);
		obj_tree.SetRowValue(obj_node, 1, Format("[%d,%d,%d,%d]", obj.bbox.left, obj.bbox.top, obj.bbox.right, obj.bbox.bottom));
		obj_tree.SetRowValue(obj_node, 2, obj.name);

		for(int j = 0; j < obj.keypoints.GetCount(); j++) {
			KeypointInstance& kp = obj.keypoints[j];
			String kp_label = kp.label + " " + GetStateText(kp.visibility_state);
			int kp_node = obj_tree.Add(obj_node, Null, kp.id, kp_label);
			kp_node_by_id.Add(kp.id, kp_node);
			RegisterNodeMeta(kp_node, NodeMeta::KEYPOINT, kp.id, obj.id);
			obj_tree.SetRowValue(kp_node, 1, Format("(%.1f, %.1f)", kp.x, kp.y));
			obj_tree.SetRowValue(kp_node, 2, kp.label);
		}
		obj_tree.Open(obj_node);
	}

	if(!entry->suggestions.IsEmpty()) {
		int sug_root = obj_tree.Add(root, Null, Null, "AI Suggestions");
		obj_tree.Open(sug_root);
		RegisterNodeMeta(sug_root, NodeMeta::UNKNOWN, -1);
		for(int i = 0; i < entry->suggestions.GetCount(); i++) {
			AnnotationObject& obj = entry->suggestions[i];
			String label = obj.name << " (conf: " << Format("%.2f", obj.score) << ")";
			int sug_node = obj_tree.Add(sug_root, Null, obj.id + 2000000, label);
			sug_node_by_id.Add(obj.id, sug_node);
			RegisterNodeMeta(sug_node, NodeMeta::SUGGESTION, obj.id);
			obj_tree.SetRowValue(sug_node, 1, Format("[%d,%d,%d,%d]", obj.bbox.left, obj.bbox.top, obj.bbox.right, obj.bbox.bottom));
			obj_tree.SetRowValue(sug_node, 2, Format("conf %.2f", obj.score));
		}
	}

	ScheduleListSync();
}

bool AnnotateView::IsSuggestion(int id) {
	if(!entry)
		return false;
	for(const auto& s : entry->suggestions)
		if(s.id == id)
			return true;
	return false;
}

void AnnotateView::ScheduleListSync() {
	if(selected_kp_id != -1 && kp_node_by_id.Find(selected_kp_id) >= 0) {
		SetTreeCursor(kp_node_by_id.Get(selected_kp_id));
		return;
	}
	int first_selected = GetFirstSelectedId();
	if(first_selected != -1 && IsSuggestion(first_selected) && sug_node_by_id.Find(first_selected) >= 0) {
		SetTreeCursor(sug_node_by_id.Get(first_selected));
		return;
	}
	if(first_selected != -1 && ann_node_by_id.Find(first_selected) >= 0)
		SetTreeCursor(ann_node_by_id.Get(first_selected));
}

void AnnotateView::SyncListSelectionFromCurrent() {
	ScheduleListSync();
}

void AnnotateView::SetTreeCursor(int node) {
	if(node < 0)
		return;
	tree_selection_suppressed = true;
	obj_tree.SetCursor(node);
	tree_selection_suppressed = false;
}

void AnnotateView::OnAcceptSuggestion() {
	if(!entry || selected_id == -1)
		return;
	if(IsSuggestion(selected_id)) {
		if(cmdmgr) {
			cmdmgr->Execute(new AcceptSuggestionCommand(*entry, selected_id));
			RefreshAfterCommand();
		} else {
			for(int i = 0; i < entry->suggestions.GetCount(); i++) {
				if(entry->suggestions[i].id == selected_id) {
					AnnotationObject obj = entry->suggestions[i];
					obj.accepted = true;
					obj.rejected = false;
					entry->annotations.Add(pick(obj));
					entry->suggestions.Remove(i);
					entry->has_annotations = true;
					break;
				}
			}
			RefreshObjectTree();
			Refresh();
		}
		WhenLog(Format("Accepted suggestion %d", selected_id));
	}
}

void AnnotateView::OnRejectSuggestion() {
	if(!entry || selected_id == -1)
		return;
	if(IsSuggestion(selected_id)) {
		if(cmdmgr) {
			cmdmgr->Execute(new RejectSuggestionCommand(*entry, selected_id));
			ClearSelectionState();
			RefreshAfterCommand();
		} else {
			for(int i = 0; i < entry->suggestions.GetCount(); i++) {
				if(entry->suggestions[i].id == selected_id) {
					AnnotationObject obj = entry->suggestions[i];
					obj.rejected = true;
					obj.accepted = false;
					entry->rejected_suggestions.Add(pick(obj));
					entry->suggestions.Remove(i);
					break;
				}
			}
			ClearSelectionState();
			RefreshObjectTree();
			Refresh();
		}
		WhenLog(Format("Rejected suggestion %d", selected_id));
	}
}

void AnnotateView::OnAcceptAll() {
	if(!entry || entry->suggestions.IsEmpty())
		return;
	if(cmdmgr) {
		while(!entry->suggestions.IsEmpty())
			cmdmgr->Execute(new AcceptSuggestionCommand(*entry, entry->suggestions[0].id));
		RefreshAfterCommand();
	} else {
		for(auto& s : entry->suggestions) {
			s.accepted = true;
			s.rejected = false;
			entry->annotations.Add(pick(s));
		}
		entry->suggestions.Clear();
		entry->has_annotations = true;
		RefreshObjectTree();
		Refresh();
	}
	WhenLog("Accepted all suggestions for image");
}

void AnnotateView::OnCategorySel() {
	if(cat_list.IsCursor() && categories) {
		int idx = cat_list.GetCursor();
		active_category_id = (*categories)[idx].id;
		WhenLog("Selected category: " + (*categories)[idx].name);
	}
}

void AnnotateView::OnCategorySettings() {
	if(cat_list.IsCursor() && categories) {
		int idx = cat_list.GetCursor();
		Category& cat = (*categories)[idx];
		CategorySettingsDialog dlg(cat);
		if(dlg.Run() == IDOK) {
			RefreshCategoryList();
			RefreshObjectTree();
			Refresh();
			WhenCategoriesChanged();
		}
	}
}

void AnnotateView::FocusObjectById(int id) {
	if(!entry)
		return;
	for(auto& obj : entry->annotations)
		if(obj.id == id) {
			PanToObject(obj);
			break;
		}
}

void AnnotateView::FocusSuggestionById(int id) {
	if(!entry)
		return;
	for(auto& obj : entry->suggestions)
		if(obj.id == id) {
			PanToObject(obj);
			break;
		}
}

void AnnotateView::OnTreeSel() {
	if(tree_selection_suppressed || !obj_tree.IsCursor())
		return;
	int node = obj_tree.GetCursor();
	int idx = node_meta.Find(node);
	if(idx < 0)
		return;
	auto meta = node_meta[idx];
	if(meta.type == NodeMeta::OBJECT) {
		selected_id = meta.id;
		selected_ids.Clear();
		selected_ids.Add(selected_id);
		selected_kp_id = -1;
		FocusObjectById(selected_id);
		WhenLog(Format("Selected object ID %d", selected_id));
	} else if(meta.type == NodeMeta::SUGGESTION) {
		selected_id = meta.id;
		selected_ids.Clear();
		selected_ids.Add(selected_id);
		selected_kp_id = -1;
		FocusSuggestionById(selected_id);
		WhenLog(Format("Selected suggestion ID %d", selected_id));
	} else if(meta.type == NodeMeta::KEYPOINT) {
		selected_kp_id = meta.id;
		selected_id = meta.extra;
		selected_ids.Clear();
		selected_ids.Add(selected_id);
		FocusObjectById(selected_id);
		WhenLog(Format("Selected keypoint ID %d", selected_kp_id));
	} else {
		return;
	}
	Refresh();
}

END_UPP_NAMESPACE
