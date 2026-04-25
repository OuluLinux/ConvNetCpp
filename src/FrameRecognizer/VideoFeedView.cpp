#include "VideoFeedView.h"

NAMESPACE_UPP


void DrawRectOutline(Draw& w, const Rect& r, Color c) {
	if(r.IsEmpty())
		return;
	w.DrawRect(r.left, r.top, r.GetWidth(), 1, c);
	w.DrawRect(r.left, r.bottom - 1, r.GetWidth(), 1, c);
	w.DrawRect(r.left, r.top, 1, r.GetHeight(), c);
	w.DrawRect(r.right - 1, r.top, 1, r.GetHeight(), c);
}

Rect FitToRect(Size src, Rect dst) {
	if(src.cx <= 0 || src.cy <= 0 || dst.GetWidth() <= 0 || dst.GetHeight() <= 0)
		return Rect(0, 0, 0, 0);
	double sx = (double)dst.GetWidth() / (double)src.cx;
	double sy = (double)dst.GetHeight() / (double)src.cy;
	double scale = min(sx, sy);
	int w = max(1, (int)floor(src.cx * scale));
	int h = max(1, (int)floor(src.cy * scale));
	int x = dst.left + (dst.GetWidth() - w) / 2;
	int y = dst.top + (dst.GetHeight() - h) / 2;
	return RectC(x, y, w, h);
}

String SlotGroupName(const String& slot_id) {
	int p = slot_id.Find('_');
	if(p <= 0)
		return slot_id;
	return slot_id.Left(p);
}

Color SlotColor(const String& slot_id) {
	static Color palette[] = {
		Color(255, 80, 80), Color(80, 200, 80), Color(80, 130, 255),
		Color(255, 200, 0), Color(200, 80, 255), Color(80, 220, 220),
		Color(255, 140, 0), Color(180, 255, 80),
	};
	String key = SlotGroupName(slot_id);
	int h = 0;
	for(int i = 0; i < key.GetCount(); i++)
		h = h * 31 + (byte)key[i];
	if(h < 0)
		h = -h;
	return palette[h % (int)(sizeof(palette) / sizeof(palette[0]))];
}

String ResultLabel(const SlotResult& r, bool show_offsets) {
	String v = !r.raw_text.IsEmpty() ? r.raw_text : r.top_class;
	if(v.IsEmpty() && r.class_index >= 0)
		v = AsString(r.class_index);
	
	String label = r.slot_id;
	if(!v.IsEmpty())
		label << ": " << v;
	
	if(show_offsets && (fabs(r.offset_dx) >= 0.5 || fabs(r.offset_dy) >= 0.5))
		label << Format(" [%+d,%+d]", (int)round(r.offset_dx), (int)round(r.offset_dy));
	
	return label;
}

VideoFeedView::VideoFeedView() {
	BackPaint();
}

VideoFeedView::~VideoFeedView() {
	Disconnect();
}

int VideoFeedView::AddFeed(const String& host, int port) {
	One<FeedSlot>& slot = feeds_.Add();
	slot.Create();
	slot->host = host;
	slot->port = port;
	int idx = feeds_.GetCount() - 1;

	RemoteCaptureSource probe(host, port);
	if(!probe.Connect()) {
		feeds_.Remove(idx);
		return -1;
	}
	probe.Disconnect();

	{
		Mutex::Lock __(slot->mutex);
		slot->running = true;
	}
	slot->thread.Run([=] { CaptureLoopFeed(idx); });
	Refresh();
	return idx;
}

void VideoFeedView::RemoveFeed(int idx) {
	if(idx < 0 || idx >= feeds_.GetCount() || !feeds_[idx])
		return;
	{
		Mutex::Lock __(feeds_[idx]->mutex);
		feeds_[idx]->running = false;
	}
	if(feeds_[idx]->thread.IsOpen())
		feeds_[idx]->thread.Wait();
	feeds_.Remove(idx);
	Refresh();
}

bool VideoFeedView::Connect(const String& host, int port) {
	Disconnect();
	return AddFeed(host, port) >= 0;
}

void VideoFeedView::Disconnect() {
	for(int i = feeds_.GetCount() - 1; i >= 0; i--)
		RemoveFeed(i);
}

void VideoFeedView::CaptureLoopFeed(int idx) {
	if(idx < 0 || idx >= feeds_.GetCount() || !feeds_[idx])
		return;
	FeedSlot& feed = *feeds_[idx];
	RemoteCaptureSource source(feed.host, feed.port);
	if(!source.Connect()) {
		Mutex::Lock __(feed.mutex);
		feed.running = false;
		return;
	}

	while(true) {
		{
			Mutex::Lock __(feed.mutex);
			if(!feed.running)
				break;
		}

		Image frame = source.GetFrame();
		if(!frame.IsEmpty()) {
			{
				Mutex::Lock __(feed.mutex);
				feed.latest_frame = frame;
				feed.last_frame_time_ms = GetTickCount();
				feed.frame_count++;
			}
			PostCallback([=] { Refresh(); });
		}
		else {
			Sleep(20);
		}
	}
	source.Disconnect();
}

Image VideoFeedView::GetLatestFrame() const {
	{
		Mutex::Lock __(static_lock_);
		if(!static_frame_.IsEmpty())
			return static_frame_;
	}
	if(feeds_.IsEmpty() || !feeds_[0])
		return Image();
	Mutex::Lock __(feeds_[0]->mutex);
	return feeds_[0]->latest_frame;
}

Image VideoFeedView::GetCurrentImage() const {
	Mutex::Lock __(static_lock_);
	return static_frame_;
}

void VideoFeedView::SetCurrentImage(const Image& img) {
	Mutex::Lock __(static_lock_);
	static_frame_ = img;
	rendered_frame_.Clear();
	has_rendered_frame_ = false;
	Refresh();
}

void VideoFeedView::SetRenderedImage(const Image& source, const Image& rendered) {
	Mutex::Lock __(static_lock_);
	static_frame_ = source;
	rendered_frame_ = rendered;
	has_rendered_frame_ = !rendered.IsEmpty();
	Refresh();
}

void VideoFeedView::SetSlotResults(const Vector<SlotResult>& results) {
	Mutex::Lock __(results_lock_);
	overlay_results_ <<= results;
	Refresh();
}

void VideoFeedView::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SBlack());

	Image frame;
	bool use_rendered = false;
	{
		Mutex::Lock __(static_lock_);
		if(has_rendered_frame_ && !rendered_frame_.IsEmpty()) {
			frame = rendered_frame_;
			use_rendered = true;
		}
		else
			frame = static_frame_;
	}
	if(frame.IsEmpty() && !feeds_.IsEmpty() && feeds_[0]) {
		Mutex::Lock __(feeds_[0]->mutex);
		frame = feeds_[0]->latest_frame;
	}
	if(frame.IsEmpty())
		return;

	Rect viewport = FitToRect(frame.GetSize(), Rect(sz));
	if(viewport.IsEmpty())
		return;

	w.DrawImage(viewport.left, viewport.top, Rescale(frame, viewport.GetSize()));

	Vector<SlotResult> results;
	{
		Mutex::Lock __(results_lock_);
		results <<= overlay_results_;
	}
	if(use_rendered)
		return;

	for(int i = 0; i < results.GetCount(); i++) {
		const SlotResult& r = results[i];
		if(r.pixel_bbox.IsEmpty())
			continue;

		double sx = (double)viewport.GetWidth() / (double)frame.GetWidth();
		double sy = (double)viewport.GetHeight() / (double)frame.GetHeight();
		Rect rr(viewport.left + int(r.pixel_bbox.left * sx),
		        viewport.top + int(r.pixel_bbox.top * sy),
		        viewport.left + int(r.pixel_bbox.right * sx),
		        viewport.top + int(r.pixel_bbox.bottom * sy));
		if(rr.IsEmpty())
			continue;

		Color c = SlotColor(r.slot_id);
		DrawRectOutline(w, rr, c);

		String label = ResultLabel(r, show_offsets);
		Font f = StdFont();
		int tw = GetTextSize(label, f).cx + 6;
		int th = f.GetCy() + 4;
		int tx = rr.left;
		int ty = max(viewport.top, rr.top - th);
		w.DrawRect(tx, ty, tw, th, Blend(c, Black(), 80));
		w.DrawText(tx + 3, ty + 2, label, f, White());
	}
}

END_UPP_NAMESPACE
