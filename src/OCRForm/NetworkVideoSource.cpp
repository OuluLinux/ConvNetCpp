#include "NetworkVideoSource.h"

#include <plugin/jpg/jpg.h>

#include "YuvConvert.h"

NAMESPACE_UPP

bool NetworkVideoSource::ParseHostPort(const String& addr, String& host, int& port) const
{
	int c = addr.ReverseFind(':');
	if(c <= 0 || c + 1 >= addr.GetCount())
		return false;

	host = TrimBoth(addr.Left(c));
	String p = TrimBoth(addr.Mid(c + 1));
	if(host.IsEmpty())
		return false;

	port = StrInt(p);
	return port > 0 && port <= 65535;
}

bool NetworkVideoSource::Open(const String& param)
{
	Close();

	String host;
	int port = 0;
	if(!ParseHostPort(param, host, port))
		return false;

	if(!sock.Connect(host, port))
		return false;

	opened = true;
	last_id = 0;
	return true;
}

void NetworkVideoSource::Close()
{
	if(sock.IsOpen())
		sock.Close();
	opened = false;
	last_id = 0;
	latest_frame.Clear();
}

bool NetworkVideoSource::GrabPacket(String& payload, uint32& packet_id)
{
	payload.Clear();
	packet_id = 0;
	if(!IsOpen())
		return false;

	if(!sock.Timeout(5000).Put(&last_id, 4))
		return false;

	uint32 resp_id = 0;
	uint32 sz = 0;
	if(!sock.Timeout(5000).GetAll(&resp_id, 4))
		return false;
	if(!sock.Timeout(5000).GetAll(&sz, 4))
		return false;

	packet_id = resp_id;
	if(sz == 0)
		return true;

	payload = sock.GetAll(sz);
	if(payload.GetCount() != (int)sz)
		return false;

	last_id = resp_id;
	return true;
}

bool NetworkVideoSource::DecodePacket(const String& payload, Image& out) const
{
	out.Clear();
	if(payload.IsEmpty())
		return true;

	if(payload.GetCount() >= 16 && memcmp(~payload, "YUV0", 4) == 0) {
		const byte* p = (const byte*)~payload;
		uint32 w = *(const uint32*)(p + 4);
		uint32 h = *(const uint32*)(p + 8);
		uint32 sz = *(const uint32*)(p + 12);
		if((int)sz <= 0 || payload.GetCount() < 16 + (int)sz || w == 0 || h == 0)
			return false;
		if((int)sz != (int)(w * h * 2))
			return false;

		ImageBuffer ib((int)w, (int)h);
		YUYVToImage(p + 16, (int)w, (int)h, ib);
		ib.SetKind(IMAGE_OPAQUE);
		out = ib;
		return true;
	}

	Image jpg = JPGRaster().LoadString(payload);
	if(jpg.IsEmpty())
		return false;
	out = jpg;
	return true;
}

Image NetworkVideoSource::GetFrame()
{
	if(!IsOpen())
		return Null;

	String payload;
	uint32 packet_id = 0;
	if(!GrabPacket(payload, packet_id)) {
		Close();
		return Null;
	}

	if(payload.IsEmpty())
		return latest_frame;

	Image decoded;
	if(!DecodePacket(payload, decoded))
		return latest_frame;

	latest_frame = decoded;
	return latest_frame;
}

END_UPP_NAMESPACE
