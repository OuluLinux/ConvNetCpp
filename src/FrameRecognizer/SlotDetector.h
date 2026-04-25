#ifndef _FrameRecognizer_SlotDetector_h_
#define _FrameRecognizer_SlotDetector_h_

#include <Core/Core.h>

NAMESPACE_UPP

struct DetectedSlot : Moveable<DetectedSlot> {
	String slot_id;
	double confidence = 0.0;
	Rect   bbox;
};

class SlotDetector {
public:
	bool Init(const String& model_path, const String& script_dir);
	void SetSlotConfOverrides(const VectorMap<String, double>& overrides);
	Vector<DetectedSlot> Detect(const String& image_path, double conf_threshold = 0.3);

private:
	String model_path_;
	String script_dir_;
	VectorMap<String, double> slot_conf_overrides_;
};

END_UPP_NAMESPACE

#endif
