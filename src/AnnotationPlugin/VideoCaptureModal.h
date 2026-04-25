#ifndef _AnnotationTool_VideoCaptureModal_h_
#define _AnnotationTool_VideoCaptureModal_h_

#include <CtrlLib/CtrlLib.h>

NAMESPACE_UPP

class VideoPreviewCtrl : public Ctrl {
public:
	typedef VideoPreviewCtrl CLASSNAME;

	void SetImage(const Image& img) { image = img; Refresh(); }
	const Image& GetImage() const   { return image; }

	virtual void Paint(Draw& w) override;

private:
	Image image;
};

class VideoCaptureModal : public TopWindow {
public:
	typedef VideoCaptureModal CLASSNAME;

	VideoCaptureModal();

	bool        ExecuteCapture();
	const Image& GetCapturedImage() const { return captured_image; }

private:
	bool   FetchFrame(Image& out);
	void   OnConnect();
	void   OnCapture();
	void   OnPoll();
	String GetAddress() const;

	Label       endpoint_label;
	EditString  endpoint;
	Button      connect;
	Button      capture;
	Button      cancel;
	Label       status;
	VideoPreviewCtrl preview;
	TimeCallback poll;
	Image       current_frame;
	Image       captured_image;
	uint32      last_id = 0;
};

END_UPP_NAMESPACE

#endif
