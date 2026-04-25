#ifndef _OCRDatasetEditor_OCRDatasetProject_h_
#define _OCRDatasetEditor_OCRDatasetProject_h_

#include <OCR/OCRTypes.h>

namespace Upp {

namespace OCR {

class OCRDatasetProject : public OCRDataset {
public:
	OCRDatasetProject();

	bool Load(const String& path);
	bool Save(const String& path);
	
	void Clear();
	void RemoveImage(int index);
	
	void SetModified(bool mod = true) { modified = mod; }
	bool IsModified() const { return modified; }

	// Extension to original OCRDataset to support settings
	void SetSetting(const String& key, const Value& val) { settings.Add(key, val); modified = true; }
	Value GetSetting(const String& key) const { int q = settings.Find(key); return q >= 0 ? settings.GetValue(q) : Value(); }
	
	String GetJson() const;
	bool SetJson(const String& json);

private:
	bool modified;
	ValueMap settings;
};

} // namespace OCR

} // namespace Upp

#endif
