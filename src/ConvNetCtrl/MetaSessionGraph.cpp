#include "ConvNetCtrl.h"

namespace ConvNet {

MetaSessionGraph::MetaSessionGraph() : legend(plot) {
	mses = NULL;
	mode = 0;
	
	plot.SetMode(PLOT_AA).SetLimits(-5,5,-5,5);
	Add(plot.SizePos());
	
	legend.SetRect(60,10,1,1); // We adjust the size later, when we know how much space it needs
	legend.SetBackground(White());
	legend.WhenSync = THISBACK(ResizeLegend);
	Add(legend);
	
	interval = 200;
	last_steps = 0;
	show_legend = true;
	
	SetModeLatestLoss();
}

void MetaSessionGraph::ResizeLegend() {
	legend.SetSize(show_legend ? legend.GetSizeHint() : Size(1,1));
}

void MetaSessionGraph::SetMetaSession(MetaSession& mses) {
	this->mses = &mses;
	mses.WhenStepInterval << THISBACK(StepInterval);
	mses.WhenFinishFold << THISBACK(PostClear);
	Clear();
}

void MetaSessionGraph::StepInterval(int num_steps) {
	// log progress to graph, (full loss)
	if (num_steps >= last_steps + interval) {
		AddValue();
		last_steps = num_steps;
	}
}

void MetaSessionGraph::RefreshData() {
	plot.Sync();
	plot.Refresh();
}

void MetaSessionGraph::AddValue() {
	ASSERT(mses);
	
	int mnet_iter = mses->GetIteration();
	int session_count = mses->GetSessionCount();
	bool fail = false;
	for(int i = 0; i < session_count && i < plot.data.GetCount(); i++) {
		const Session& ses = mses->GetSession(i);
		double d;
		if (mode == MODE_LATESTLOSS) {
			const Window& win = ses.GetAccuracyWindow();
			if (win.GetBufferCount() == 0) {fail = true; break;}
			d = win.GetLatest();
		}
		else if (mode == MODE_LOSS) {
			const Window& win = ses.GetLossWindow();
			if (win.GetBufferCount() == 0) {fail = true; break;}
			d = win.GetAverage();
		}
		else if (mode == MODE_TRAINACC) {
			const Window& win = ses.GetAccuracyWindow();
			if (win.GetBufferCount() == 0) {fail = true; break;}
			d = win.GetAverage();
		}
		else if (mode == MODE_TESTACC) {
			const Window& win = ses.GetTestingAccuracyWindow();
			if (win.GetBufferCount() == 0) {fail = true; break;}
			d = win.GetAverage();
		}
		else return;
		plot.data[i].AddXY(mnet_iter, d);
		
		if (d > max) max = d;
		if (d < min) min = d;
	}
	
	if (!fail) {
		plot.SetLimits(plot.data[0][1].x, mnet_iter, min, max);
		plot.SetModify();
	}
	
	PostCallback(THISBACK(RefreshData));
}

void MetaSessionGraph::Clear() {
	int count = mses->GetSessionCount();
	
	max = -DBL_MAX;
	min = +DBL_MAX;
	
	plot.data.SetCount(count);
	for(int i = 0; i < count; i++) {
		Session& ses = mses->GetSession(i);
		plot.data[i].SetCount(0);
		plot.data[i]
			.SetTitle(ses.GetTrainer()->GetKey())
			.SetThickness(1.0)
			.SetColor(Rainbow((double)i / count))
			.AddXY(0,0);
	}
	plot.SetLimits(-1, 1, -1, 1);
	plot.SetModify();
	ResizeLegend();
}


}
