// VideoServer protocol: TCP on host:port (default 8082). Client sends 4-byte
// uint32 last_id, server replies with 4-byte uint32 frame_id, 4-byte uint32
// jpeg_size, then jpeg_size bytes of JPEG payload for the newest frame when
// frame_id > last_id.
#include <plugin/jpg/jpg.h>

#include "VideoCaptureModal.h"

NAMESPACE_UPP

void VideoPreviewCtrl::Paint(Draw& w)
{
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	if(image.IsEmpty()) {
		w.DrawText(12, 12, "No frame loaded", StdFont().Bold(), SColorDisabled());
		return;
	}

	Image scaled = Rescale(image, sz);
	w.DrawImage(0, 0, scaled);
}

VideoCaptureModal::VideoCaptureModal()
{
	Title("Capture From VideoServer");
	Sizeable().Zoomable();
	SetRect(0, 0, 520, 420);

	Add(endpoint_label.SetLabel("host:port").LeftPosZ(12, 70).TopPosZ(12, 20));
	endpoint.SetText("127.0.0.1:8082");
	Add(endpoint.LeftPosZ(86, 180).TopPosZ(10, 24));
	Add(connect.SetLabel("Connect").LeftPosZ(278, 90).TopPosZ(10, 24));
	Add(capture.SetLabel("Capture Frame").LeftPosZ(370, 120).TopPosZ(10, 24));
	Add(preview.HSizePosZ(12, 12).VSizePosZ(46, 60));
	Add(status.HSizePosZ(12, 12).BottomPosZ(36, 20));
	Add(cancel.SetLabel("Cancel").RightPosZ(12, 100).BottomPosZ(8, 24));

	status.SetLabel("VideoServer source checked: current implementation uses a raw TCP JPEG protocol, not an HTTP snapshot endpoint.");
	connect << [this] { OnConnect(); };
	capture << [this] { OnCapture(); };
	cancel << [this] { captured_image.Clear(); Breaker(IDCANCEL); };
	capture.Disable();
}

String VideoCaptureModal::GetAddress() const
{
	String addr = TrimBoth(endpoint.GetData().ToString());
	return addr.IsEmpty() ? String("127.0.0.1:8082") : addr;
}

bool VideoCaptureModal::FetchFrame(Image& out)
{
	Vector<String> parts = Split(GetAddress(), ':');
	if(parts.GetCount() != 2)
		return false;

	String host = TrimBoth(parts[0]);
	int port = StrInt(TrimBoth(parts[1]));
	if(host.IsEmpty() || port <= 0)
		return false;

	TcpSocket s;
	if(!s.Connect(host, port))
		return false;

	if(!s.Timeout(1000).Put(&last_id, 4))
		return false;

	uint32 id = 0;
	uint32 sz = 0;
	if(!s.Timeout(1000).GetAll(&id, 4))
		return false;
	if(!s.Timeout(1000).GetAll(&sz, 4))
		return false;
	if(sz == 0)
		return false;

	String payload = s.Timeout(1500).GetAll((int)sz);
	if(payload.GetCount() != (int)sz)
		return false;

	Image img = StreamRaster::LoadStringAny(payload);
	if(img.IsEmpty())
		return false;

	last_id = id;
	out = img;
	return true;
}

void VideoCaptureModal::OnConnect()
{
	Image img;
	if(!FetchFrame(img)) {
		status.SetLabel("Connect failed. Current VideoServer source exposes raw TCP frame polling on host:port.");
		return;
	}

	current_frame = img;
	preview.SetImage(current_frame);
	capture.Enable();
	status.SetLabel("Connected using raw TCP polling. Preview refresh runs every 200 ms.");
	poll.Set(-200, [this] { OnPoll(); });
}

void VideoCaptureModal::OnPoll()
{
	Image img;
	if(FetchFrame(img)) {
		current_frame = img;
		preview.SetImage(current_frame);
	}
	poll.Set(-200, [this] { OnPoll(); });
}

void VideoCaptureModal::OnCapture()
{
	if(current_frame.IsEmpty()) {
		status.SetLabel("No frame available to capture.");
		return;
	}
	captured_image = current_frame;
	Breaker(IDOK);
}

bool VideoCaptureModal::ExecuteCapture()
{
	captured_image.Clear();
	current_frame.Clear();
	preview.SetImage(Null);
	last_id = 0;
	int rc = Run();
	poll.Kill();
	return rc == IDOK && !captured_image.IsEmpty();
}

END_UPP_NAMESPACE
