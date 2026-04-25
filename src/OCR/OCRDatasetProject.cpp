#include "OCRDatasetProject.h"

namespace Upp {

namespace OCR {

OCRDatasetProject::OCRDatasetProject() {
	modified = false;
}

void OCRDatasetProject::Clear() {
	OCRDataset::Clear();
	settings.Clear();
	modified = false;
}

void OCRDatasetProject::RemoveImage(int index) {
	OCRDataset::RemoveImage(index);
	modified = true;
}

bool OCRDatasetProject::Load(const String& path) {
	String json = LoadFile(path);
	if (json.IsEmpty()) return false;
	
	if (!SetJson(json)) return false;
	
	modified = false;
	return true;
}

bool OCRDatasetProject::Save(const String& path) {
	String json = GetJson();
	if (SaveFile(path, json)) {
		modified = false;
		return true;
	}
	return false;
}

String OCRDatasetProject::GetJson() const {
	Json j;
	j("version", "1.0")
	 ("created", GetSysTime());
	
	JsonArray imgs;
	for (int i = 0; i < GetCount(); i++) {
		Json img_map;
		img_map("path", GetImagePath(i));
		
		JsonArray segs;
		const Vector<OCRSegment>& src_segs = GetSegments(i);
		for (const OCRSegment& s : src_segs) {
			Json seg_map;
			seg_map("bounds", JsonArray() << s.bounds.left << s.bounds.top << s.bounds.right << s.bounds.bottom)
			       ("text", s.text);
			
			JsonArray chars;
			for (const OCRChar& c : s.chars) {
				Json char_map;
				char_map("bounds", JsonArray() << c.bounds.left << c.bounds.top << c.bounds.right << c.bounds.bottom)
				        ("label", c.label);
				chars << char_map;
			}
			seg_map("chars", chars);
			segs << seg_map;
		}
		img_map("segments", segs);
		imgs << img_map;
	}
	j("images", imgs);
	j("settings", (Value)settings);
	
	return j;
}

bool OCRDatasetProject::SetJson(const String& json) {
	Value j = ParseJSON(json);
	if (j.IsNull()) return false;
	
	Clear();
	
	Value imgs = j["images"];
	for (int i = 0; i < imgs.GetCount(); i++) {
		Value img_map = imgs[i];
		String path = img_map["path"];
		
		// Attempt to load image from path
		Image img;
		FileIn in(path);
		if (in.IsOpen()) img = StreamRaster::LoadAny(in);
		AddImage(img, path);
		
		Value segs = img_map["segments"];
		for (int k = 0; k < segs.GetCount(); k++) {
			Value seg_map = segs[k];
			OCRSegment s;
			
			Value b = seg_map["bounds"];
			if (b.GetCount() == 4)
				s.bounds = Rect((int)b[0], (int)b[1], (int)b[2], (int)b[3]);
			
			s.text = seg_map["text"];
			
			Value chars = seg_map["chars"];
			for (int m = 0; m < chars.GetCount(); m++) {
				Value char_map = chars[m];
				OCRChar c;
				
				Value cb = char_map["bounds"];
				if (cb.GetCount() == 4)
					c.bounds = Rect((int)cb[0], (int)cb[1], (int)cb[2], (int)cb[3]);
				
				c.label = char_map["label"];
				s.chars.Add(c);
			}
			AddSegment(i, s);
		}
	}
	
	Value s = j["settings"];
	if (s.Is<ValueMap>())
		settings = s;
	else
		settings.Clear();
	
	return true;
}

} // namespace OCR

} // namespace Upp
