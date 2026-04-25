#ifndef _OCRForm_NetworkVideoSource_h_
#define _OCRForm_NetworkVideoSource_h_

#include <Core/Core.h>

#include "IVideoSource.h"

NAMESPACE_UPP

class NetworkVideoSource : public IVideoSource {
public:
	virtual String GetName() const override { return "VideoServer"; }
	virtual String GetTypeID() const override { return "network"; }

	virtual bool Open(const String& param) override;
	virtual void Close() override;
	virtual bool IsOpen() const override { return opened && sock.IsOpen() && !sock.IsError(); }

	virtual Image GetFrame() override;
	virtual int   GetPreferredPollMs() const override { return 33; }

private:
	bool ParseHostPort(const String& addr, String& host, int& port) const;
	bool GrabPacket(String& payload, uint32& packet_id);
	bool DecodePacket(const String& payload, Image& out) const;

	TcpSocket sock;
	bool      opened = false;
	uint32    last_id = 0;
	Image     latest_frame;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::NetworkVideoSource;
}

#endif
