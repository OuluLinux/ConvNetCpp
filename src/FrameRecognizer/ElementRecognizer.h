#ifndef _FrameRecognizer_CardRecognizer_h_
#define _FrameRecognizer_CardRecognizer_h_

#include <Core/Core.h>

#include "SlotDetector.h"

NAMESPACE_UPP

struct CardResult : Moveable<CardResult> {
	String slot_id;
	String level;
	String category;
};

class CardRecognizer {
public:
	bool Init(const String& script_dir, const String& templates_dir);
	Vector<CardResult> Recognize(const String& image_path, const Vector<DetectedSlot>& card_slots);

private:
	String script_dir_;
	String templates_dir_;
};

END_UPP_NAMESPACE

#endif
