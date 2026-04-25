#include "VideoSourceRegistry.h"

NAMESPACE_UPP

VideoSourceRegistry& VideoSourceRegistry::Get()
{
	static VideoSourceRegistry registry;
	return registry;
}

void VideoSourceRegistry::Register(const String& type_id, const String& name, Factory factory)
{
	if(type_id.IsEmpty() || !factory)
		return;

	int q = names.Find(type_id);
	if(q >= 0) {
		names[q] = name;
		factories[q] = pick(factory);
		return;
	}

	names.Add(type_id, name);
	factories.Add(pick(factory));
}

IVideoSource* VideoSourceRegistry::Create(const String& type_id) const
{
	int q = names.Find(type_id);
	if(q < 0)
		return NULL;
	return factories[q] ? factories[q]() : NULL;
}

Vector<String> VideoSourceRegistry::GetTypeIDs() const
{
	Vector<String> ids;
	for(int i = 0; i < names.GetCount(); i++)
		ids.Add(names.GetKey(i));
	return ids;
}

String VideoSourceRegistry::GetName(const String& type_id) const
{
	int q = names.Find(type_id);
	return q >= 0 ? names[q] : String();
}

bool VideoSourceRegistry::IsRegistered(const String& type_id) const
{
	return names.Find(type_id) >= 0;
}

END_UPP_NAMESPACE
