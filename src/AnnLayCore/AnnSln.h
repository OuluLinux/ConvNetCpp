#ifndef _AnnLayCore_AnnSln_h_
#define _AnnLayCore_AnnSln_h_

#include <Core/Core.h>

NAMESPACE_UPP

struct CvTemplateSubCrop {
	int x = 0, y = 0, w = 0, h = 0;

	void Jsonize(JsonIO& jio) {
		jio("x", x)("y", y)("w", w)("h", h);
	}
};

struct CvTemplateGroup {
	String            name;                   // group key (e.g. "board_card")
	Vector<String>    slot_stems;
	String            templates_label_a_dir;  // resolved absolute path after load
	String            templates_label_b_dir;  // resolved absolute path after load
	CvTemplateSubCrop label_a_crop;
	CvTemplateSubCrop label_b_crop;
	String            label_a_display = "label_a";
	String            label_b_display = "label_b";
	String            present_display = "present";
	String            match_method = "TM_CCOEFF_NORMED";
	double            match_threshold = 0.65;
	double            zoom_min = 0.7, zoom_max = 1.3;
	int               zoom_steps = 7;
	Index<String>     rotation_slots;
	VectorMap<String, double> slot_rotation_deg; // optional fixed rotation per slot stem
	double            rotation_min_deg = -4, rotation_max_deg = 4;
	int               rotation_steps = 9;

	CvTemplateGroup() = default;
	CvTemplateGroup(const CvTemplateGroup& s);
	CvTemplateGroup& operator=(const CvTemplateGroup& s);

	void Jsonize(JsonIO& jio);
};

struct AnnSln {
	int    version = 1;
	String name = "new_project";
	String annprj;
	String annlay;
	String mlui;
	String crops_dir;
	String templates_dir;
	String images_dir;
	String recognition_script;
	VectorMap<String, String> model_sets;
	int    train_epochs = 50;
	int    pass2_min_verified = 100;
	Array<CvTemplateGroup> cv_template_groups;

	AnnSln() {}
	AnnSln(const AnnSln& s);
	AnnSln& operator=(const AnnSln& s);

	void Jsonize(JsonIO& jio);

	bool Load(const String& path);
	bool Save(const String& path) const;
};

END_UPP_NAMESPACE

#endif
