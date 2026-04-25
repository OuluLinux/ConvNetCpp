#ifndef _FrameRecognizer_CaptureSource_h_
#define _FrameRecognizer_CaptureSource_h_

#include <Core/Core.h>
#include <Draw/Draw.h>

NAMESPACE_UPP

class CaptureSource {
public:
	virtual ~CaptureSource() {}
	virtual bool Connect() = 0;
	virtual void Disconnect() = 0;
	virtual Image GetFrame() = 0; // returns Null if no new frame
	virtual bool IsConnected() const = 0;
};

END_UPP_NAMESPACE

#endif
