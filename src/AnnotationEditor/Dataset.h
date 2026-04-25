#ifndef _AnnotationEditor_Dataset_h_
#define _AnnotationEditor_Dataset_h_

#include <Core/Core.h>
#include <Core/MluiScript/MluiScript.h>

NAMESPACE_UPP

struct Category : Moveable<Category> {
	int id = 0;
	String name;
	String slot_id;
	String supercategory;
	Color color = Null;
	Vector<String> keypoint_labels;
	Vector<String> keypoint_connects_to;

	Category() {}
	Category(const Category& c);
	Category& operator=(const Category& c);

	void Jsonize(JsonIO& jio);
};

struct KeypointInstance : Moveable<KeypointInstance> {
	int id = 0;
	String label;
	double x = 0, y = 0;
	int visibility_state = 0;

	void Jsonize(JsonIO& jio);
};

struct AnnotationObject : Moveable<AnnotationObject> {
	int id = 0;
	int category_id = -1;
	String name;
	String slot_id;
	Color color = Null;
	bool visible = true;
	int label_visibility_state = 2; // 2 = Labeled visible
	Vector< Vector<Pointf> > polygons;
	Vector<KeypointInstance> keypoints;
	VectorMap<String,String> metadata;

	double score = 1.0; // Confidence score for AI suggestions
	bool accepted = false;
	bool rejected = false;
	Rectf bbox = Rectf(0, 0, 0, 0);

	void UpdateBBox();

	AnnotationObject() {}
	AnnotationObject(const AnnotationObject& o);
	AnnotationObject& operator=(const AnnotationObject& o);

	// Expand a [x,y,w,h] bbox shorthand into a 4-point rectangle polygon.
	// Called on load when "bbox" is present and polygons is empty.
	void BBoxToPolygon(double x, double y, double w, double h);

	void Jsonize(JsonIO& jio);
};

struct ImageEntry : Moveable<ImageEntry> {
	String file_path;
	String file_name;
	int width = 0;
	int height = 0;
	bool has_annotations = false;
	bool reviewed = false;
	double priority_score = 0.0;
	Vector<AnnotationObject> annotations;
	Vector<AnnotationObject> suggestions;
	Vector<AnnotationObject> rejected_suggestions;
	Vector<MluiScriptLink> mlui_scripts; // .mlui files associated with this image

	String table_currency_unit = "BB";
	int dealer = -1;
	double pot_total = 0.0;
	double round_pot = 0.0;
	double side_pot = 0.0;
	Vector<int> player_in_table;
	Vector<int> player_in_game;
	String hero_cards;
	String board_cards;
	VectorMap<String, String> image_metadata;

	static bool ParseBoolMeta(const String& s, bool def = false) {
		String v = ToLower(TrimBoth(s));
		if(v == "1" || v == "true" || v == "yes" || v == "on") return true;
		if(v == "0" || v == "false" || v == "no" || v == "off") return false;
		return def;
	}
	static int ParseIntMeta(const String& s, int def = 0) {
		if(IsNull(s) || s.IsEmpty()) return def;
		return StrInt(s);
	}
	static double ParseDoubleMeta(const String& s, double def = 0.0) {
		if(IsNull(s) || s.IsEmpty()) return def;
		return StrDbl(s);
	}

	void SyncLegacyToImageMetadata();
	void SyncImageMetadataToLegacy();
	void EnsureMetadataDefaults();

	ImageEntry() { EnsureMetadataDefaults(); }
	ImageEntry(const ImageEntry& e);
	ImageEntry& operator=(const ImageEntry& e);

	void Jsonize(JsonIO& jio);
};

struct Dataset : Moveable<Dataset> {
	String name;
	String slot_id;
	String folder_path;
	Vector<String> default_categories;
	String created_by = "User";
	int image_count = 0;
	Vector<ImageEntry> images;
	String mlui_script_path; // default .mlui for all images in this dataset (optional)

	Dataset() {}
	Dataset(const Dataset& d);
	Dataset& operator=(const Dataset& d);

	void Jsonize(JsonIO& jio);
};

struct Project {
	String format = "AnnotationEditorProject";
	int version = 1;
	Time last_saved;
	bool is_autosave = false;
	String autosave_source_project_path;

	Vector<Category> categories;
	Vector<Dataset> datasets;

	int last_dataset_index = -1;
	int last_image_index = -1;
	String last_mlui_script_path;

	void Jsonize(JsonIO& jio);
};

END_UPP_NAMESPACE

#endif
