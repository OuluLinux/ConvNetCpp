#ifndef _OCRForm_ScreenCaptureSource_h_
#define _OCRForm_ScreenCaptureSource_h_

#include <CtrlCore/CtrlCore.h>

#include "IVideoSource.h"

NAMESPACE_UPP

class ScreenCaptureSource : public IVideoSource {
public:
	virtual String GetName() const override { return "Screen Capture"; }
	virtual String GetTypeID() const override { return "screen"; }

	virtual bool Open(const String& param) override;
	virtual void Close() override;
	virtual bool IsOpen() const override { return open; }

	virtual Image GetFrame() override;
	virtual int   GetPreferredPollMs() const override { return 100; }

private:
	bool ParseRect(const String& param, Rect& out) const;

	bool open = false;
	Rect capture_rect;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::ScreenCaptureSource;
}

#endif
