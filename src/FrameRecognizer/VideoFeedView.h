#ifndef _FrameRecognizer_VideoFeedView_h_
#define _FrameRecognizer_VideoFeedView_h_

#include <CtrlLib/CtrlLib.h>

#include "RemoteCaptureSource.h"
#include <AnnLayCore/AnchoredSlotRecognizer.h>

NAMESPACE_UPP

class VideoFeedView : public Ctrl {
public:
	typedef VideoFeedView CLASSNAME;

	struct FeedSlot {
		String  host;
		int     port = 8082;
		Thread  thread;
		mutable Mutex mutex;
		Image   latest_frame;
		int64   last_frame_time_ms = 0;
		int     frame_count = 0;
		bool    running = false;
	};

	VideoFeedView();
	virtual ~VideoFeedView();

	bool Connect(const String& host, int port);
	void Disconnect();
	int  AddFeed(const String& host, int port);
	void RemoveFeed(int idx);
	int  GetFeedCount() const { return feeds_.GetCount(); }

	Image GetLatestFrame() const;
	Image GetCurrentImage() const;
	void  SetCurrentImage(const Image& img);
	void  SetRenderedImage(const Image& source, const Image& rendered);
	void  SetSlotResults(const Vector<SlotResult>& results);

	void  ShowOffsets(bool b = true) { show_offsets = b; Refresh(); }
	bool  IsShowingOffsets() const { return show_offsets; }

	virtual void Paint(Draw& w) override;

private:
	void CaptureLoopFeed(int idx);

	Vector<One<FeedSlot>> feeds_;

	mutable Mutex static_lock_;
	Image         static_frame_;
	Image         rendered_frame_;
	bool          has_rendered_frame_ = false;

	bool          show_offsets = true;

	mutable Mutex      results_lock_;
	Vector<SlotResult> overlay_results_;
};

END_UPP_NAMESPACE

#endif
