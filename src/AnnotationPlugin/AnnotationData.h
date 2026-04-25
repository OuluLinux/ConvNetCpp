#ifndef _AnnotationTool_AnnotationData_h_
#define _AnnotationTool_AnnotationData_h_

#include <Core/Core.h>

NAMESPACE_UPP

struct AnnBBox : Moveable<AnnBBox> {
	int x = 0;
	int y = 0;
	int w = 0;
	int h = 0;

	Rect ToRect() const { return RectC(x, y, max(0, w), max(0, h)); }
};

struct AnnObject : Moveable<AnnObject> {
	int    id = 0;
	int    image_id = 0;
	int    category_id = 0;
	AnnBBox bbox;
	String label;
	String notes;
};

struct AnnImage : Moveable<AnnImage> {
	int    id = 0;
	String filename;
	int    width = 0;
	int    height = 0;
};

struct AnnCategory : Moveable<AnnCategory> {
	int    id = 0;
	String name;
	String supercategory;
};

struct AnnDataset : Moveable<AnnDataset> {
	Vector<AnnImage>    images;
	Vector<AnnCategory> categories;
	Vector<AnnObject>   annotations;
};

struct AnnVideoSource : Moveable<AnnVideoSource> {
	String label;
	String host = "127.0.0.1";
	int    port = 18082;
	AnnBBox window_rect;
};

inline Vector<String> GetInitialAnnotationTaxonomy()
{
	Vector<String> out;
	out
		<< "seat_name"
		<< "seat_stack"
		<< "seat_action"
		<< "pot_label"
		<< "board_card"
		<< "board_area"
		<< "dealer_button"
		<< "hero_indicator"
		<< "window_rect";
	return out;
}

inline Vector<AnnCategory> GetInitialAnnotationCategories()
{
	Vector<String> names = GetInitialAnnotationTaxonomy();
	Vector<AnnCategory> out;
	for(int i = 0; i < names.GetCount(); i++) {
		AnnCategory& c = out.Add();
		c.id = i + 1;
		c.name = names[i];
		c.supercategory = "gui";
	}
	return out;
}

END_UPP_NAMESPACE

#endif
