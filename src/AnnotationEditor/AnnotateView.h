#ifndef _AnnotationEditor_AnnotateView_h_
#define _AnnotationEditor_AnnotateView_h_

#include "AnnotationEditorCommon.h"

NAMESPACE_UPP
class AnnotateView : public Ctrl {
public:
	typedef AnnotateView CLASSNAME;
	
	enum ToolType {
		TOOL_SELECT,
		TOOL_BBOX,
		TOOL_POLYGON,
		TOOL_BRUSH,
		TOOL_ERASER,
		TOOL_KEYPOINT,
		TOOL_MAGICWAND,
		TOOL_REVIEW
	};
	enum { kMaxPlayerControls = 8 };
	struct ImageMetaField : Moveable<ImageMetaField> {
		String key;
		String label;
		String type;
		String default_value;
	};
	
	AnnotateView();
	
	void SetCommandManager(CommandManager& m) { cmdmgr = &m; }

	// Insert a fully constructed AnnotationObject (id must already be set).
	void DeleteObjectById(int id);
	void AddAnnotationObject(AnnotationObject&& obj);

	void OnUndo();
	void OnRedo();
	String GetLastUndoName();
	String GetLastRedoName();
	void RefreshAfterCommand();
	void SetCategories(Vector<Category>& c);
	void SetDataset(Dataset& ds, int img_idx);
	void SetMluiScript(const MluiScript& script);
	void ClearMluiScript();
	bool ParseSeatFromSlotId(const String& sid, int& seat_zero_based, String* suffix = nullptr) const;
	bool TryParseWizardBool(const String& raw, bool& out) const;
	Color AdjustColorBrightness(Color c, int delta) const;
	Color SeatBaseColor(int seat_zero_based) const;
	bool GetSeatFieldColor(const String& key, Color& out) const;
	AttrText MakeMetaFieldAttrText(const ImageMetaField& f) const;
	AttrText MakeBoolAttrText(bool value) const;
	void UpdateMetadataModeUI();
	void ShowMetadataWizardEditor(const String& type, const String& value);
	void ShowMetadataWizardField();
	bool CommitMetadataWizardField(bool advance);
	void OnMetadataWizardEnter();
	void OnToggleMetadataWizardMode();
	void RefreshMetadataSchemaFromMlui();

	void RefreshImageMetadataUI();
	void CommitImageMetadataFromUI();
	void OnEnterReviewMode();
	void JumpToNextIssue();

	bool HasIssue(const AnnotationObject& obj);
	void CenterOnObject(const AnnotationObject& obj);
	void PanToObject(const AnnotationObject& obj);
	void PrevImage();
	void NextImage();
	void JumpToBestImage();
	void ShowTemporaryFeedback(const String& text);
	void RefreshCategoryList();
	String GetStateText(int state);

	struct NodeMeta { enum Type { UNKNOWN, CATEGORY, OBJECT, SUGGESTION, KEYPOINT } type; int id; int extra; };

	void RegisterNodeMeta(int node, NodeMeta::Type type, int id, int extra = -1);

	void RefreshObjectTree();
	bool IsSuggestion(int id);
	void ScheduleListSync();
	void SyncListSelectionFromCurrent();
	void SetTreeCursor(int node);
	void OnAcceptSuggestion();
	void OnRejectSuggestion();
	void OnAcceptAll();
	void OnCategorySel();
	void OnCategorySettings();
	void FocusObjectById(int id);
	void FocusSuggestionById(int id);
	void OnTreeSel();
	
	void OnToggleVisibility();
	void OnGeneralSettings();
	void ClearSelectionState();
	void RemoveSelectionId(int id);
	bool DeleteSelectedObjects(bool ask_confirm = true);
	void OnGeneralDelete();
	void OnKeypointSettings();
	void OnKeypointDelete();

	void OnObjectSettings();
	
	// ---------------------------------------------------------------
	// Set geometry without drawing
	// ---------------------------------------------------------------

	// Returns the next available annotation id (max existing + 1).
	// IDs are per-image (not globally unique across the dataset).
	int GetNextObjectId() const;

	// Opens SetGeometryDialog for the currently selected annotation.
	// If no annotation is selected, creates a new one.
	void OnSetGeometry();

	// ---------------------------------------------------------------
	// MLUI Script support
	// ---------------------------------------------------------------

	// Apply a MluiScript to the current image: creates AnnotationObject stubs
	// for each slot not already present (matched by mlui_slot_id metadata).
	// Returns the number of objects added.
	int ApplyMluiScript(const MluiScript& script);

	// Populate normalized bbox_hints from current annotations back into a script.
	// Matches by slot_id stored in annotation metadata.
	// Updates script in-place; returns number of slots updated.
	int UpdateScriptHintsFromImage(MluiScript& script);

	// Expose current annotations and image info to the script editor
	// so "Copy hints from image" works.
	void PopulateScriptEditorReference(MluiScriptEditor& editor);
	void OnObjectDelete();
	void OnCopyAnnotations();

	void CenterImage();
	
	Pointf ScreenToImage(Point p) { return Pointf((p.x - offset.x) / zoom, (p.y - offset.y) / zoom); }
	Point ImageToScreen(Pointf p) { return Point(int(p.x * zoom + offset.x), int(p.y * zoom + offset.y)); }
	Rectf GetViewportRect();
	void UpdateToolLabel();
	void InvalidateScaledImage();
	void EnsureScaledImage();

	virtual void Paint(Draw& w) override;
	
	void DrawAnnotations(Draw& w);

	void DrawDashLine(Draw& w, Point p1, Point p2, int thick, Color c);
	
	double GetCategoryAcceptanceRate(int cat_id);

	void SetDatasetsPtr(const Vector<Dataset>& ds) { datasets_ptr = &ds; }

	void DrawHoverInfo(Draw& w);
	
	virtual void LeftDown(Point p, dword keyflags) override;
	virtual void LeftDouble(Point p, dword keyflags) override;
	virtual void MouseMove(Point p, dword keyflags) override;
	virtual void LeftUp(Point p, dword keyflags) override;
	bool CancelActiveInteractions();
	virtual void MiddleDown(Point p, dword keyflags) override;
	virtual void MiddleUp(Point p, dword keyflags) override;
	virtual void RightDown(Point p, dword keyflags) override;
	virtual void MouseWheel(Point p, int zdelta, dword keyflags) override;
	
	void ClosePolygon();
	void CreateAutoObject();
	void AddBBox(Pointf start, Pointf end);
	void OnPlaceKeypoint(Pointf pt);
	void OnMagicWand(Point p);
	void DuplicateObject();
	void CopyAnnotations();
	void PasteAnnotations();
	void CycleVisibility();
	void SelectCategory(int index);
	void SetActiveCategoryById(int category_id);
	
	int    GetSelectedId()        const { return selected_id; }
	int    GetFirstSelectedId()   const { return selected_ids.GetCount() > 0 ? selected_ids[0] : -1; }
	int    GetActiveCategoryId()  const { return active_category_id; }
	int    GetCurrentImageIndex() const { return current_img_idx; }
	double GetZoom()              const { return zoom; }
	String GetImagePath()         const { return entry ? ResolveImagePath(entry->file_path) : String(); }
	int    GetAnnotationCount()   const { return entry ? entry->annotations.GetCount() : 0; }
	int    GetSuggestionCount()   const { return entry ? entry->suggestions.GetCount() : 0; }
	int    GetDatasetImageCount() const { return dataset ? dataset->images.GetCount() : 0; }
	bool   CanPrev()              const { return dataset && dataset->images.GetCount() > 1; }
	bool   CanNext()              const { return dataset && dataset->images.GetCount() > 1; }
	int    GetImageWidth()        const { return img ? img.GetWidth()  : 0; }
	int    GetImageHeight()       const { return img ? img.GetHeight() : 0; }
	ImageEntry* GetCurrentEntry() const { return entry; }
	void SetCurrentTool(ToolType tool);
	void UpdateToolButtons();

	String GetToolName() const {
		switch(current_tool) {
			case TOOL_SELECT:    return "select";
			case TOOL_BBOX:      return "bbox";
			case TOOL_POLYGON:   return "polygon";
			case TOOL_BRUSH:     return "brush";
			case TOOL_ERASER:    return "eraser";
			case TOOL_MAGICWAND: return "wand";
			case TOOL_KEYPOINT:  return "keypoint";
			case TOOL_REVIEW:    return "review";
			default:             return "select";
		}
	}

	void SetToolByName(const String& name);
	void SelectById(int id);
	void SelectAllObjects();

	// Returns list of script slot_ids not yet present as annotations in current image.
	Vector<String> GetUnfilledSlots(const MluiScript& script) const;

	// Called after external bulk edits to ImageEntry to resync the view.
	void RefreshAfterBulkEdit(ImageEntry& modified_entry);

	virtual bool Key(dword key, int count) override;
	
		Callback WhenBack; Event<String> WhenLog; Event<> WhenCategoriesChanged; Event<> WhenCommandExecuted; Event<int> WhenOpenImage; Event<> WhenDirty;
		Event<int> WhenBBoxAdded;
		Function<void(const AnnotationObject&)> WhenCopyHintToLastMluiScript;
		Function<String(const String&)> WhenResolveImagePath;

		int selected_id = -1, selected_kp_id = -1;
		Index<int> selected_ids;
		ParentCtrl& GetCategoriesPanel() { return categories_panel; }
		ParentCtrl& GetObjectsPanel() { return objects_panel; }
		ParentCtrl& GetSettingsPanel() { return settings_panel; }
		ParentCtrl& GetMetadataPanel() { return metadata_panel; }

private:
	String ResolveImagePath(const String& path) const;
		ParentCtrl toolbar; Button btn_back, btn_prev, btn_next, btn_close_poly, btn_copy, btn_center, btn_undo, btn_redo;
		ButtonOption btn_select, btn_bbox, btn_poly, btn_brush, btn_eraser, btn_wand, btn_keypoint, btn_review;
		Label lbl_active_tool;
		ParentCtrl categories_panel, objects_panel, settings_panel, metadata_panel;
		Label lbl_cat_title, lbl_tree_title, lbl_wand_settings, lbl_threshold, lbl_view_settings, lbl_flow_settings;
		Label lbl_meta_title, lbl_currency, lbl_dealer, lbl_pot_total, lbl_round_pot, lbl_side_pot, lbl_hero_cards, lbl_board_cards, lbl_players_title;
		Label lbl_player_row[kMaxPlayerControls];
		ArrayCtrl cat_list;
		Button btn_cat_settings;
		EditInt edit_threshold, edit_dealer;
		EditDouble edit_pot_total, edit_round_pot, edit_side_pot;
		EditString edit_currency, edit_hero_cards, edit_board_cards;
		Label lbl_meta_schema_info, lbl_meta_wizard_field, lbl_meta_wizard_hint;
		ArrayCtrl meta_schema_table;
		ButtonOption btn_meta_wizard;
		EditString edit_meta_wizard_string, edit_meta_wizard_bool;
		EditIntSpin edit_meta_wizard_int;
		EditDoubleSpin edit_meta_wizard_double;
		Option chk_show_hover, chk_auto_create, chk_auto_advance;
		Option opt_player_in_table[kMaxPlayerControls], opt_player_in_game[kMaxPlayerControls];
		TreeArrayCtrl obj_tree;
		Button btn_obj_visible, btn_obj_settings, btn_obj_delete, btn_obj_geometry, btn_obj_add;
		VectorMap<int, NodeMeta> node_meta;
		bool tree_selection_suppressed = false;
		bool metadata_syncing_ = false;
		bool metadata_wizard_mode_ = false;
		int metadata_wizard_index_ = -1;
		Vector<ImageMetaField> meta_fields_;
		MluiScript mlui_script_;
		VectorMap<int, int> ann_node_by_id, sug_node_by_id, kp_node_by_id;
	
	Dataset* dataset = nullptr; int current_img_idx = -1; ImageEntry* entry = nullptr; Vector<Category>* categories = nullptr; const Vector<Dataset>* datasets_ptr = nullptr;
		Image img; double zoom = 1.0; Pointf offset = Pointf(0, 0); Pointf mouse_pos_img = Pointf(0, 0); ToolType current_tool = TOOL_SELECT; int active_category_id = -1; bool drawing_bbox = false; Pointf bbox_start, bbox_current; bool selecting_rect = false; Pointf select_rect_start, select_rect_current; bool select_rect_toggle_mode = false, select_rect_add_mode = false; Vector<Pointf> current_poly; bool panning = false; Point pan_start; Pointf offset_start; bool dragging_pt = false; int drag_obj_idx = -1, drag_poly_idx = -1, drag_pt_idx = -1; Vector<Pointf> drag_poly_before; bool moving_objects = false; Pointf move_start_img; Vector<int> move_obj_indices; Vector<Vector<Vector<Pointf>>> move_old_polys; Vector<Vector<KeypointInstance>> move_old_kps; CommandManager* cmdmgr = nullptr;
		bool brushing = false; Vector<Vector<Pointf>> brush_strokes; Pointf last_brush_pos; double brush_radius = 20.0;
		bool dragging_kp = false; Pointf kp_old_pos;
		int hovered_id = -1, hovered_kp_id = -1; bool show_hover_info = true; Point last_mouse_pos;
		AnnotationObject clipboard_object;
		Image scaled_img;
		Size scaled_size = Size(0, 0);
		double scaled_zoom = -1;
		String temp_overlay_text; int temp_overlay_timeout = 0;
};

// =========================================================================
// Dataset UI
// =========================================================================

END_UPP_NAMESPACE

#endif
