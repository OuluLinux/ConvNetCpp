#include "ConvNetCtrl.h"

namespace ConvNet {

TrainingGraph::TrainingGraph() {
	ses = NULL;
	mode = MODE_REWARD;
	
	plotter.SetMode(PLOT_AA).SetLimits(-5,5,-5,5);
	plotter.SetBackground(IsDarkTheme() ? AdjustIfDark(White()) : White());
	plotter.SetFrameColor(IsDarkTheme() ? AdjustIfDark(White()) : White());
	plotter.SetFontColor(IsDarkTheme() ? AdjustIfDark(Black()) : Black());
	plotter.SetAxisColor(IsDarkTheme() ? AdjustIfDark(Black()) : Black());
	plotter.data.Add();
	plotter.data.Add();
	plotter.data[0].SetTitle("Reward").SetThickness(1.5).SetColor(AdjustIfDark(Color(220, 60, 60)));
	plotter.data[1].SetDash("1.5").SetTitle("Average").SetThickness(1.0).SetColor(AdjustIfDark(Color(60, 100, 220)));
	
	Add(plotter.SizePos());
	
	average_size = 20;
	interval = 200;
	update_period_ms = 200;
	last_add_ms = 0;
	last_steps = 0;
	limit = 0;
}

void TrainingGraph::SetSession(Session& ses) {
	this->ses = &ses;
	ses.WhenStepInterval << THISBACK(StepInterval);
}

void TrainingGraph::RefreshData() {
	plotter.Sync();
	plotter.Refresh();
}

void TrainingGraph::StepInterval(int num_steps) {
	int64 now = GetTickCount();
	if(last_add_ms > 0 && (now - last_add_ms) < update_period_ms)
		return;

	// log progress to graph, (full loss)
	if (num_steps >= last_steps + interval) {
		AddValue();
		last_steps = num_steps;
		last_add_ms = now;
	}
}

void TrainingGraph::Clear() {
	plotter.data[0].Clear();
	plotter.data[1].Clear();
	last_steps = 0;
	last_add_ms = 0;
}

void TrainingGraph::AddValue() {
	if (!ses) return;
	
	double av;
	
	if (mode == MODE_REWARD) {
		av = ses->GetRewardAverage();
	}
	else if (mode == MODE_LOSS) {
		double xa = ses->GetLossAverage();
		double xw = ses->GetL2DecayLossAverage();
		if (xa < 0) return;
		av = xa + (xw >= 0 ? xw : 0);
	}
	
	AddValue(av);
}

void TrainingGraph::AddValue(double value) {
	if (!IsFin(value)) return;
	int c = plotter.data[0].GetCount();
	int id = c;
	if (id) id = plotter.data[0][id-1].x+1;
	
	plotter.data[0].AddXY(id, value);
	int count = id + 1;
	if (count < 2) return;
	
	int pos = c-1;
	double sum = 0;
	int av_count = 0;
	for(int i = 0; i < average_size && pos >= 0; i++) {
		sum += plotter.data[0][pos].y;
		av_count++;
		pos--;
	}
	double avav = sum / av_count;
	plotter.data[1].AddXY(id, avav);
	
	if (limit > 0) {
		while (plotter.data[0].GetCount() > limit) {
			plotter.data[0].Remove(0);
			plotter.data[1].Remove(0);
		}
	}
	
	double min = +DBL_MAX;
	double max = -DBL_MAX;
	c = plotter.data[0].GetCount();
	for(int i = 0; i < c; i++) {
		double d = plotter.data[0][i].y;
		if (d > max) max = d;
		if (d < min) min = d;
	}
	double diff = max - min;
	if (diff <= 0) return;
	double center = min + diff / 2;
	plotter.SetLimits(plotter.data[0][0].x, id, min, max);
	plotter.SetModify();
	
	PostCallback(THISBACK(RefreshData));
}

}
