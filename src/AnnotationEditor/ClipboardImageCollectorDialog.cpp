#include "ClipboardImageCollectorDialog.h"

NAMESPACE_UPP

ClipboardImageCollectorDialog::ClipboardImageCollectorDialog() {
	Title("Clipboard Image Collector");
	Sizeable();
	SetRect(0, 0, 560, 180);

	Add(lbl_title.SetLabel("Polling clipboard every second. Unique images are appended to active dataset.")
		.HSizePos(10, 10).TopPos(10, 20));
	lbl_title.SetFont(StdFont().Bold());
	Add(lbl_status.SetLabel("Status: waiting for clipboard image...").HSizePos(10, 10).TopPos(40, 20));
	Add(lbl_counts.SetLabel("Added: 0  Skipped: 0").HSizePos(10, 10).TopPos(65, 20));
	Add(lbl_last_saved.SetLabel("Last saved: -").HSizePos(10, 10).TopPos(90, 20));
	Add(btn_close.SetLabel("Close").RightPos(10, 100).BottomPos(10, 28));
	btn_close << [=] { Close(); };
	SetTimeCallback(-1000, THISBACK(PollClipboard), 1);
}

ClipboardImageCollectorDialog::~ClipboardImageCollectorDialog() {
	KillTimeCallback(1);
}

void ClipboardImageCollectorDialog::PollClipboard() {
	Image img = ReadClipboardImage();
	if(img.IsEmpty())
		return;
	String png = PNGEncoder().SaveString(img);
	if(png.IsEmpty()) {
		lbl_status.SetLabel("Status: clipboard had image, but PNG encoding failed.");
		return;
	}
	String signature = Format("%08x_%d", (int)GetHashValue(png), png.GetCount());
	if(signature == last_clip_signature)
		return;
	last_clip_signature = signature;
	if(seen_signatures.Find(signature) >= 0) {
		skipped_count++;
		lbl_status.SetLabel("Status: duplicate clipboard image ignored.");
		UpdateCounters();
		return;
	}
	seen_signatures.FindAdd(signature);

	String saved_path;
	bool ok = WhenCaptureImage && WhenCaptureImage(img, signature, &saved_path);
	if(ok) {
		added_count++;
		lbl_status.SetLabel("Status: image added to dataset.");
		lbl_last_saved.SetLabel("Last saved: " + saved_path);
	}
	else {
		skipped_count++;
		lbl_status.SetLabel("Status: image skipped (already present or save failed).");
	}
	UpdateCounters();
}

void ClipboardImageCollectorDialog::UpdateCounters() {
	lbl_counts.SetLabel(Format("Added: %d  Skipped: %d", added_count, skipped_count));
}

END_UPP_NAMESPACE
