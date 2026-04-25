#include "StaticImageSource.h"

NAMESPACE_UPP

bool StaticImageSource::Open(const String& param)
{
	Close();
	if(param.IsEmpty())
		return false;

	Image img = StreamRaster::LoadFileAny(param);
	if(IsNull(img))
		return false;

	frame = img;
	source_path = param;
	return true;
}

void StaticImageSource::Close()
{
	frame.Clear();
	source_path.Clear();
}

END_UPP_NAMESPACE
