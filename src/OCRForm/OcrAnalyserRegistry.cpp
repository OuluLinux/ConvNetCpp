#include "OcrAnalyserRegistry.h"

NAMESPACE_UPP

namespace {

bool IsEmptyAnnotation(const OcrAnnotation& a)
{
	return a.rect.IsEmpty()
	    && a.widget_type.IsEmpty()
	    && a.name.IsEmpty()
	    && a.children.IsEmpty()
	    && a.analyser_id.IsEmpty()
	    && a.confidence == 0.0
	    && a.hints.IsEmpty();
}

}

OcrAnalyserRegistry& OcrAnalyserRegistry::Get()
{
	static OcrAnalyserRegistry registry;
	return registry;
}

void OcrAnalyserRegistry::Register(IOcrAnalyser& analyser)
{
	String id = analyser.GetID();
	if(id.IsEmpty())
		return;

	RWMutex::WriteLock __(lock);
	for(int i = 0; i < analysers.GetCount(); i++) {
		if(analysers[i] == &analyser || analysers[i]->GetID() == id) {
			analysers[i] = &analyser;
			return;
		}
	}
	analysers.Add(&analyser);
}

void OcrAnalyserRegistry::Unregister(const String& id)
{
	if(id.IsEmpty())
		return;

	RWMutex::WriteLock __(lock);
	for(int i = analysers.GetCount() - 1; i >= 0; i--)
		if(analysers[i] && analysers[i]->GetID() == id)
			analysers.Remove(i);
}

IOcrAnalyser* OcrAnalyserRegistry::Find(const String& id)
{
	if(id.IsEmpty())
		return NULL;

	RWMutex::ReadLock __(lock);
	for(int i = 0; i < analysers.GetCount(); i++)
		if(analysers[i] && analysers[i]->GetID() == id)
			return analysers[i];
	return NULL;
}

const Vector<IOcrAnalyser*>& OcrAnalyserRegistry::GetAll() const
{
	return analysers;
}

Vector<OcrAnnotation> OcrAnalyserRegistry::AnalyseAll(const OcrAnalyserContext& ctx)
{
	Vector<IOcrAnalyser*> active;
	{
		RWMutex::ReadLock __(lock);
		active = clone(analysers);
	}

	Vector<OcrAnnotation> result;
	for(int i = 0; i < active.GetCount(); i++) {
		IOcrAnalyser* analyser = active[i];
		if(!analyser || !analyser->CanAnalyse(ctx))
			continue;

		OcrAnnotation proposal = analyser->Analyse(ctx);
		if(IsEmptyAnnotation(proposal))
			continue;
		result.Add(proposal);
	}

	for(int i = 0; i < result.GetCount(); i++)
		for(int j = i + 1; j < result.GetCount(); j++)
			if(result[j].confidence > result[i].confidence)
				Swap(result[i], result[j]);

	return result;
}

END_UPP_NAMESPACE
