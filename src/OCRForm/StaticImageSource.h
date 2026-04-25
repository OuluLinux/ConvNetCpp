#ifndef _OCRForm_StaticImageSource_h_
#define _OCRForm_StaticImageSource_h_

#include "IVideoSource.h"

NAMESPACE_UPP

class StaticImageSource : public IVideoSource {
public:
	virtual String GetName() const override { return "Static Image"; }
	virtual String GetTypeID() const override { return "static_image"; }

	virtual bool Open(const String& param) override;
	virtual void Close() override;
	virtual bool IsOpen() const override { return !IsNull(frame); }

	virtual Image GetFrame() override { return frame; }
	virtual int   GetPreferredPollMs() const override { return 0; }

private:
	Image  frame;
	String source_path;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::StaticImageSource;
}

#endif
