#include "ReinforcedLearning.h"

World::World() {
	clock = 0;
	
	// Add agent
	RLAgent& agent = agents.Add();
	agent.world = this;
	
	double value = 0;
	for(int i = 0; i < 1000; i++) {
		
		int count = Random(8);
		int diff = Random(2) * 2 - 1;
		
		for(int j = 0; j < count; j++) {
			value += diff;
			buffer.Add(value);
		}
		value -= diff;
		buffer.Add(value);
	}
}

void World::Tick() {
	// tick the environment
	clock++;
	
	
	// let the agents behave in the world based on their input
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		agents[i].Forward();
	}
	
	// agents are given the opportunity to learn based on feedback of their action on environment
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		agents[i].Backward();
	}
}

void World::Paint(Draw& d) {
	
	Size sz = GetSize();
	
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	
	int max_count = buffer.GetCount();
	double min = +DBL_MAX;
	double max = -DBL_MAX;
	double last = 0.0;
	double peak = -DBL_MAX;
	
	int max_steps = 0;
	int count = buffer.GetCount();
	if (max_count > 0)
		count = Upp::min(count, max_count);
	for(int j = 0; j < count; j++) {
		double d = buffer[j];
		if (d > max) max = d;
		if (d < min) min = d;
	}
	if (count > max_steps)
		max_steps = count;
	
	
	if (max_steps > 1 && max >= min) {
		double diff = max - min;
		double xstep = (double)sz.cx / (max_steps - 1);
		Font fnt = Monospace(10);
		
		if (count >= 2) {
			polyline.SetCount(0);
			for(int j = 0; j < count; j++) {
				double v = buffer[j];
				last = v;
				int x = (int)(j * xstep);
				int y = (int)(sz.cy - (v - min) / diff * sz.cy);
				polyline.Add(Point(x, y));
				if (v > peak) peak = v;
			}
			if (polyline.GetCount() >= 2)
				id.DrawPolyline(polyline, 1, Color(81, 145, 137));
		}
		
		{
			int y = 0;
			String str = DblStr(peak);
			Size str_sz = GetTextSize(str, fnt);
			id.DrawRect(16, y, str_sz.cx, str_sz.cy, White());
			id.DrawText(16, y, str, fnt, Black());
		}
		{
			int y = 0;
			String str = DblStr(last);
			Size str_sz = GetTextSize(str, fnt);
			id.DrawRect(sz.cx - 16 - str_sz.cx, y, str_sz.cx, str_sz.cy, White());
			id.DrawText(sz.cx - 16 - str_sz.cx, y, str, fnt, Black());
		}
		{
			int y = (int)(sz.cy - (zero_line - min) / diff * sz.cy);
			id.DrawLine(0, y, sz.cx, y, 1, Black());
			if (zero_line != 0.0) {
				int y = sz.cy - 10;
				String str = DblStr(zero_line);
				Size str_sz = GetTextSize(str, fnt);
				id.DrawRect(16, y, str_sz.cx, str_sz.cy, White());
				id.DrawText(16, y, str, fnt, Black());
			}
		}
	}
	
	d.DrawImage(0, 0, id);
}
