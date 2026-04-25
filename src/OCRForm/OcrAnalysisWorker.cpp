#include "OcrAnalysisWorker.h"

#include <CtrlCore/CtrlCore.h>

#include "OcrAnalyserRegistry.h"

NAMESPACE_UPP

namespace {

Rect NormalizeWorkerRect(const Rect& rect)
{
	int left = min(rect.left, rect.right);
	int top = min(rect.top, rect.bottom);
	int right = max(rect.left, rect.right);
	int bottom = max(rect.top, rect.bottom);
	if(right <= left)
		right = left + 1;
	if(bottom <= top)
		bottom = top + 1;
	return Rect(left, top, right, bottom);
}

String BuildWorkerAnnotationPath(const String& prefix, const OcrAnnotation& annotation, int index)
{
	String label = annotation.name;
	if(label.IsEmpty())
		label = annotation.widget_type;
	if(label.IsEmpty())
		label = Format("annotation_%d", index);
	return prefix.IsEmpty() ? label : prefix + "/" + label;
}

bool IsWorkerMeaningfulProposal(const OcrAnnotation& current, const OcrAnnotation& proposal)
{
	if(!proposal.children.IsEmpty())
		return true;
	if(!proposal.widget_type.IsEmpty() && proposal.widget_type != current.widget_type)
		return true;
	if(!proposal.name.IsEmpty() && proposal.name != current.name)
		return true;
	if(!proposal.hints.IsEmpty())
		return true;
	if(!proposal.rect.IsEmpty() && NormalizeWorkerRect(proposal.rect) != NormalizeWorkerRect(current.rect))
		return true;
	return false;
}

double ResolveProposalConfidence(const OcrAnnotation& proposal)
{
	return max(0.0, proposal.confidence);
}

bool RegionTouchesChanged(const Rect& region, const Vector<Rect>& changed_regions)
{
	for(int i = 0; i < changed_regions.GetCount(); i++) {
		Rect overlap = region & changed_regions[i];
		if(!overlap.IsEmpty())
			return true;
	}
	return false;
}

void CollectRecursive(const Image& frame,
                      const OcrFormLayout& layout,
                      const VectorMap<String, String>& global_hints,
                      const Vector<Rect>& changed_regions,
                      const OcrAnnotation* parent,
                      const Array<OcrAnnotation>& annotations,
                      const String& path_prefix,
                      VectorMap<String, AnnotationProposal>& out,
                      Index<String>& visited)
{
	Vector<int> process_order;
	process_order.Reserve(annotations.GetCount());
	if(changed_regions.IsEmpty()) {
		for(int i = 0; i < annotations.GetCount(); i++)
			process_order.Add(i);
	}
	else {
		Vector<int> delayed;
		delayed.Reserve(annotations.GetCount());
		for(int i = 0; i < annotations.GetCount(); i++) {
			Rect region = NormalizeWorkerRect(annotations[i].rect);
			if(!region.IsEmpty() && RegionTouchesChanged(region, changed_regions))
				process_order.Add(i);
			else
				delayed.Add(i);
		}
		for(int i = 0; i < delayed.GetCount(); i++)
			process_order.Add(delayed[i]);
	}

	for(int k = 0; k < process_order.GetCount(); k++) {
		int i = process_order[k];
		const OcrAnnotation& ann = annotations[i];
		String path = BuildWorkerAnnotationPath(path_prefix, ann, i);
		visited.FindAdd(path);

		Rect region = NormalizeWorkerRect(ann.rect);
		if(!region.IsEmpty()) {
			OcrAnalyserContext ctx{frame, region, parent, layout, global_hints};
			Vector<OcrAnnotation> proposals = OcrAnalyserRegistry::Get().AnalyseAll(ctx);
			for(int j = 0; j < proposals.GetCount(); j++) {
				const OcrAnnotation& p = proposals[j];
				if(p.analyser_id.IsEmpty() || p.analyser_id == "manual")
					continue;
				if(!IsWorkerMeaningfulProposal(ann, p))
					continue;

				AnnotationProposal& slot = out.GetAdd(path);
				slot.analyser_id = p.analyser_id;
				slot.proposed = p;
				slot.confidence = ResolveProposalConfidence(p);
				break;
			}
		}

		if(ann.children.GetCount())
			CollectRecursive(frame, layout, global_hints, changed_regions,
			                 &ann, ann.children, path, out, visited);
	}
}

}

OcrAnalysisWorker::OcrAnalysisWorker()
{
}

OcrAnalysisWorker::~OcrAnalysisWorker()
{
	Stop();
}

void OcrAnalysisWorker::Start()
{
	Mutex::Lock __(lock);
	if(thread_running)
		return;
	stop_flag = false;
	thread_running = thread.Run([this] { ThreadMain(); }, true);
}

void OcrAnalysisWorker::Stop()
{
	bool wait = false;
	{
		Mutex::Lock __(lock);
		if(thread_running) {
			stop_flag = true;
			cv.Broadcast();
			wait = true;
		}
	}
	if(wait) {
		thread.Wait();
		Mutex::Lock __(lock);
		thread_running = false;
		stop_flag = false;
		queue.Clear();
	}
}

void OcrAnalysisWorker::SubmitJob(AnalysisJob job)
{
	Mutex::Lock __(lock);
	if(!thread_running || stop_flag)
		return;
	queue.Add(pick(job));
	cv.Signal();
}

void OcrAnalysisWorker::ClearQueue()
{
	Mutex::Lock __(lock);
	queue.Clear();
}

bool OcrAnalysisWorker::IsRunning() const
{
	Mutex::Lock __(lock);
	return thread_running && !stop_flag;
}

void OcrAnalysisWorker::ThreadMain()
{
	for(;;) {
		AnalysisJob job;
		{
			Mutex::Lock __(lock);
			while(!stop_flag && queue.IsEmpty())
				cv.Wait(lock);
			if(stop_flag)
				break;
			job = pick(queue[0]);
			queue.Remove(0);
		}

		AnalysisResult result;
		result.id = job.id;
		result.generation = job.generation;

		Index<String> visited;
		CollectRecursive(job.frame,
		                 job.layout_snapshot,
		                 job.global_hints,
		                 job.changed_regions,
		                 NULL,
		                 job.layout_snapshot.roots,
		                 String(),
		                 result.proposals,
		                 visited);
		result.visited_paths.SetCount(visited.GetCount());
		for(int i = 0; i < visited.GetCount(); i++)
			result.visited_paths[i] = visited[i];

		Function<void(const AnalysisResult&)> cb;
		{
			Mutex::Lock __(lock);
			cb = WhenProposalsReady;
			if(stop_flag)
				continue;
		}
		if(cb)
			PostCallback([cb, result = pick(result)] { cb(result); });
	}
}

END_UPP_NAMESPACE
