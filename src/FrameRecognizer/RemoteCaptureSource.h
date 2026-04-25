#ifndef _FrameRecognizer_RemoteCaptureSource_h_
#define _FrameRecognizer_RemoteCaptureSource_h_

#include <Core/Core.h>

#include "CaptureSource.h"

NAMESPACE_UPP

class RemoteCaptureSource : public CaptureSource {
public:
	RemoteCaptureSource();
	RemoteCaptureSource(const String& host, int port);

	void SetEndpoint(const String& host, int port);

	virtual bool Connect() override;
	virtual void Disconnect() override;
	virtual Image GetFrame() override;
	virtual bool IsConnected() const override;

private:
	bool GrabPayload(String& payload, uint32& frame_id);

	String host_ = "127.0.0.1";
	int    port_ = 8082;
	bool   opened_ = false;
	uint32 last_id_ = 0;
	TcpSocket sock_;
};

END_UPP_NAMESPACE

#endif
