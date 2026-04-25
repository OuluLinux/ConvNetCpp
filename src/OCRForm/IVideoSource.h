#ifndef _OCRForm_IVideoSource_h_
#define _OCRForm_IVideoSource_h_

#include <Draw/Draw.h>

NAMESPACE_UPP

class IVideoSource {
public:
	virtual ~IVideoSource() {}

	virtual String GetName() const = 0;
	virtual String GetTypeID() const = 0;

	virtual bool Open(const String& param) = 0;
	virtual void Close() = 0;
	virtual bool IsOpen() const = 0;

	virtual Image GetFrame() = 0;
	virtual int   GetPreferredPollMs() const { return 33; }

	Event<> WhenFrameReady;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::IVideoSource;
}

#endif
