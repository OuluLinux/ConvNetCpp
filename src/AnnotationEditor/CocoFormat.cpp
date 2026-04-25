#include "CocoFormat.h"

double ComputePolygonArea(const Vector<Pointf>& poly) {
	if(poly.GetCount() < 3) return 0;
	double area = 0;
	for(int i = 0; i < poly.GetCount(); i++) {
		int j = (i + 1) % poly.GetCount();
		area += poly[i].x * poly[j].y;
		area -= poly[j].x * poly[i].y;
	}
	return abs(area) / 2.0;
}

Rectf ComputeAnnotationBBox(const AnnotationObject& obj) {
	if(obj.polygons.IsEmpty()) return Rectf(0, 0, 0, 0);
	double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
	bool found = false;
	for(const auto& poly : obj.polygons) {
		for(const auto& p : poly) {
			if(p.x < min_x) min_x = p.x;
			if(p.x > max_x) max_x = p.x;
			if(p.y < min_y) min_y = p.y;
			if(p.y > max_y) max_y = p.y;
			found = true;
		}
	}
	if(!found) return Rectf(0, 0, 0, 0);
	return Rectf(min_x, min_y, max_x, max_y);
}

double ComputeAnnotationArea(const AnnotationObject& obj) {
	double total = 0;
	for(const auto& poly : obj.polygons)
		total += ComputePolygonArea(poly);
	return total;
}

bool CocoExporter::Export(const Dataset& ds, const Vector<Category>& categories, const String& path) {
	ValueMap coco;
	
	// Info
	ValueMap info;
	info.Add("description", "Exported from AnnotationEditor");
	info.Add("year", (int)GetSysTime().year);
	info.Add("date_created", FormatTime(GetSysTime(), "YYYY-MM-DD"));
	coco.Add("info", info);
	
	// Licenses
	ValueArray licenses;
	ValueMap lic;
	lic.Add("id", 1);
	lic.Add("name", "Unknown");
	lic.Add("url", "");
	licenses.Add(lic);
	coco.Add("licenses", licenses);
	
	// Categories
	ValueArray cats;
	for(int i = 0; i < categories.GetCount(); i++) {
		ValueMap c;
		c.Add("id", categories[i].id);
		c.Add("name", categories[i].name);
		c.Add("supercategory", categories[i].supercategory);
		if(!categories[i].keypoint_labels.IsEmpty()) {
			ValueArray labels;
			for(const auto& l : categories[i].keypoint_labels) labels.Add(l);
			c.Add("keypoints", labels);
		}
		cats.Add(c);
	}
	coco.Add("categories", cats);
	
	// Images & Annotations
	ValueArray images;
	ValueArray annots;
	
	int img_id_counter = 1;
	int ann_id_counter = 1;
	
	for(int i = 0; i < ds.images.GetCount(); i++) {
		const ImageEntry& ie = ds.images[i];
		int img_id = img_id_counter++;
		
		ValueMap img;
		img.Add("id", img_id);
		img.Add("file_name", ie.file_name);
		img.Add("width", ie.width);
		img.Add("height", ie.height);
		images.Add(img);
		
		for(const auto& obj : ie.annotations) {
			ValueMap ann;
			ann.Add("id", ann_id_counter++);
			ann.Add("image_id", img_id);
			ann.Add("category_id", obj.category_id);
			ann.Add("iscrowd", 0);
			
			Rectf bbox = ComputeAnnotationBBox(obj);
			ValueArray bb;
			bb.Add(bbox.left);
			bb.Add(bbox.top);
			bb.Add(bbox.Width());
			bb.Add(bbox.Height());
			ann.Add("bbox", bb);
			
			ann.Add("area", ComputeAnnotationArea(obj));
			
			ValueArray segs;
			for(const auto& poly : obj.polygons) {
				ValueArray p;
				for(const auto& pt : poly) {
					p.Add(pt.x);
					p.Add(pt.y);
				}
				segs.Add(p);
			}
			ann.Add("segmentation", segs);
			
			annots.Add(ann);
		}
	}
	
	coco.Add("images", images);
	coco.Add("annotations", annots);
	
	String out = StoreAsJson(coco, true);
	return SaveFile(path, out);
}

bool CocoImporter::Import(Dataset& ds, Vector<Category>& categories, const String& path) {
	String data = LoadFile(path);
	if(data.IsEmpty()) return false;
	
	Value coco = ParseJSON(data);
	if(IsNull(coco) || !IsValueMap(coco)) return false;
	
	// Import categories
	Value cats = coco["categories"];
	if(IsValueArray(cats)) {
		for(int i = 0; i < cats.GetCount(); i++) {
			Value c = cats[i];
			int id = c["id"];
			String name = c["name"];
			
			bool found = false;
			for(int j = 0; j < categories.GetCount(); j++) {
				if(categories[j].id == id) { found = true; break; }
			}
			if(!found) {
				Category& nc = categories.Add();
				nc.id = id;
				nc.name = name;
				nc.supercategory = c["supercategory"];
				nc.color = Color(rand()%200, rand()%200, rand()%200);
			}
		}
	}
	
	// Map image IDs to file names
	VectorMap<int, String> img_id_map;
	Value images = coco["images"];
	if(IsValueArray(images)) {
		for(int i = 0; i < images.GetCount(); i++) {
			Value img = images[i];
			img_id_map.Add(img["id"], img["file_name"]);
		}
	}
	
	// Import annotations
	Value annots = coco["annotations"];
	if(IsValueArray(annots)) {
		for(int i = 0; i < annots.GetCount(); i++) {
			Value ann = annots[i];
			int img_id = ann["image_id"];
			if(img_id_map.Find(img_id) < 0) continue;
			String fname = img_id_map.Get(img_id);
			
			// Find image in dataset
			ImageEntry* target_ie = nullptr;
			for(int j = 0; j < ds.images.GetCount(); j++) {
				if(GetFileName(ds.images[j].file_name) == GetFileName(fname)) {
					target_ie = &ds.images[j];
					break;
				}
			}
			
			if(!target_ie) continue;
			
			AnnotationObject& obj = target_ie->annotations.Add();
			obj.id = ann["id"];
			obj.category_id = ann["category_id"];
			obj.visible = true;
			obj.label_visibility_state = 2;
			
			Value seg = ann["segmentation"];
			if(IsValueArray(seg)) {
				for(int j = 0; j < seg.GetCount(); j++) {
					Value poly_val = seg[j];
					if(IsValueArray(poly_val)) {
						Vector<Pointf>& p = obj.polygons.Add();
						for(int k = 0; k < poly_val.GetCount() - 1; k += 2) {
							p.Add(Pointf(poly_val[k], poly_val[k+1]));
						}
					}
				}
			}
			
			if(obj.polygons.IsEmpty()) {
				Value bb = ann["bbox"];
				if(IsValueArray(bb) && bb.GetCount() == 4) {
					double x = bb[0], y = bb[1], w = bb[2], h = bb[3];
					Vector<Pointf>& p = obj.polygons.Add();
					p.Add(Pointf(x, y));
					p.Add(Pointf(x + w, y));
					p.Add(Pointf(x + w, y + h));
					p.Add(Pointf(x, y + h));
				}
			}
			
			for(int j = 0; j < categories.GetCount(); j++) {
				if(categories[j].id == obj.category_id) {
					obj.name = categories[j].name + " " + AsString(target_ie->annotations.GetCount());
					obj.color = categories[j].color;
					break;
				}
			}
			
			target_ie->has_annotations = true;
		}
	}
	
	return true;
}
