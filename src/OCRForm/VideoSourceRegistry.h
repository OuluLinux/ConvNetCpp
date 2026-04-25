#ifndef _OCRForm_VideoSourceRegistry_h_
#define _OCRForm_VideoSourceRegistry_h_

#include <Core/Core.h>

#include "IVideoSource.h"

NAMESPACE_UPP

class VideoSourceRegistry {
public:
	typedef Function<IVideoSource*()> Factory;

	static VideoSourceRegistry& Get();

	void Register(const String& type_id, const String& name, Factory factory);
	IVideoSource* Create(const String& type_id) const;

	Vector<String> GetTypeIDs() const;
	String         GetName(const String& type_id) const;
	bool           IsRegistered(const String& type_id) const;

private:
	VectorMap<String, String> names;
	Vector<Factory>           factories;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::VideoSourceRegistry;
}

#endif
