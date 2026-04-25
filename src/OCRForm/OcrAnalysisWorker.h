#ifndef _OCRForm_OcrAnalysisWorker_h_
#define _OCRForm_OcrAnalysisWorker_h_

#include <Core/Core.h>
#include <Draw/Draw.h>

#include "AnnotationProposal.h"
#include "OcrFormLayout.h"

NAMESPACE_UPP

struct AnalysisJob : Moveable<AnalysisJob> {
	int                    id = 0;
	int                    generation = 0;
	Image                  frame;
	OcrFormLayout          layout_snapshot;
	VectorMap<String, String> global_hints;
	Vector<Rect>           changed_regions;
};

struct AnalysisResult : Moveable<AnalysisResult> {
	int                       id = 0;
	int                       generation = 0;
	VectorMap<String, AnnotationProposal> proposals;
	Vector<String>            visited_paths;
};

class OcrAnalysisWorker {
public:
	Function<void(const AnalysisResult&)> WhenProposalsReady;

	OcrAnalysisWorker();
	~OcrAnalysisWorker();

	void Start();
	void Stop();
	void SubmitJob(AnalysisJob job);
	void ClearQueue();
	bool IsRunning() const;

private:
	void ThreadMain();

	mutable Mutex        lock;
	ConditionVariable    cv;
	Thread               thread;
	bool                 thread_running = false;
	bool                 stop_flag = false;
	Vector<AnalysisJob>  queue;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::AnalysisJob;
using Upp::AnalysisResult;
using Upp::OcrAnalysisWorker;
}

#endif
