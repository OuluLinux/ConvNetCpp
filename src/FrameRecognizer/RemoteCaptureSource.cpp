#include "RemoteCaptureSource.h"

#include <plugin/jpg/jpg.h>

NAMESPACE_UPP

RemoteCaptureSource::RemoteCaptureSource()
{
}

RemoteCaptureSource::RemoteCaptureSource(const String& host, int port)
	: host_(host), port_(port)
{
}

void RemoteCaptureSource::SetEndpoint(const String& host, int port)
{
	host_ = host;
	port_ = port;
}

bool RemoteCaptureSource::Connect()
{
	Disconnect();
	if(!sock_.Connect(host_, port_))
		return false;
	opened_ = true;
	last_id_ = 0;
	return true;
}

void RemoteCaptureSource::Disconnect()
{
	if(sock_.IsOpen())
		sock_.Close();
	opened_ = false;
	last_id_ = 0;
}

bool RemoteCaptureSource::IsConnected() const
{
	return opened_ && sock_.IsOpen() && !sock_.IsError();
}

bool RemoteCaptureSource::GrabPayload(String& payload, uint32& frame_id)
{
	payload.Clear();
	frame_id = 0;
	if(!IsConnected())
		return false;

	if(!sock_.Timeout(5000).Put(&last_id_, 4))
		return false;

	uint32 id = 0;
	uint32 sz = 0;
	if(!sock_.Timeout(5000).GetAll(&id, 4))
		return false;
	if(!sock_.Timeout(5000).GetAll(&sz, 4))
		return false;

	frame_id = id;
	if(sz == 0)
		return true;

	payload = sock_.GetAll(sz);
	if(payload.GetCount() != (int)sz)
		return false;

	last_id_ = id;
	return true;
}

Image RemoteCaptureSource::GetFrame()
{
	String payload;
	uint32 frame_id = 0;
	if(!GrabPayload(payload, frame_id)) {
		Disconnect();
		return Null;
	}
	if(payload.IsEmpty())
		return Null;

	Image jpg = JPGRaster().LoadString(payload);
	if(jpg.IsEmpty())
		return Null;
	return jpg;
}

END_UPP_NAMESPACE
