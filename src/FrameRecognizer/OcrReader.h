#ifndef _FrameRecognizer_OcrReader_h_
#define _FrameRecognizer_OcrReader_h_

#include <Core/Core.h>

#include "SlotDetector.h"

NAMESPACE_UPP

struct OcrResult : Moveable<OcrResult> {
	String slot_id;
	String text;
	double value = 0.0; // NaN if not numeric
};

class OcrReader {
public:
	bool Init(const String& script_dir);
	Vector<OcrResult> Read(const String& image_path, const Vector<DetectedSlot>& slots);

private:
	String script_dir_;
};

END_UPP_NAMESPACE

#endif
