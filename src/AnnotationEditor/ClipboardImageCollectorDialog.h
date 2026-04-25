#ifndef _AnnotationEditor_ClipboardImageCollectorDialog_h_
#define _AnnotationEditor_ClipboardImageCollectorDialog_h_

#include "AnnotationEditorCommon.h"

NAMESPACE_UPP
class ClipboardImageCollectorDialog : public TopWindow {
public:
	typedef ClipboardImageCollectorDialog CLASSNAME;

	ClipboardImageCollectorDialog();
	~ClipboardImageCollectorDialog();

	Function<bool(const Image&, const String&, String*)> WhenCaptureImage;

private:
	void PollClipboard();
	void UpdateCounters();

	Label lbl_title, lbl_status, lbl_counts, lbl_last_saved;
	Button btn_close;
	String last_clip_signature;
	Index<String> seen_signatures;
	int added_count = 0;
	int skipped_count = 0;
};

END_UPP_NAMESPACE

#endif
