#include "RegressionPainter.h"

#define IMAGECLASS RegressionPainterImg
#define IMAGEFILE <RegressionPainter/RegressionPainter.iml>
#include <Draw/iml_source.h>

RegressionPainter::RegressionPainter() {
	Title("Image \"Painting\"");
	Icon(RegressionPainterImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	
	t =		"[\n"
			"\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":2},\n" // 2 inputs: x, y
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"regression\", \"neuron_count\":3},\n" // 3 outputs: r,g,b
			"\t{\"type\":\"sgd\", \"learning_rate\":0.01, \"momentum\":0.9, \"batch_size\":5, \"l2_decay\":0.0}\n"
			"]\n";
	
	
	img_list.AddColumn("Image").SetDisplay(Single<PainterImage>());
	img_list.SetLineCy(96);
	for(int i = 0; i < 19; i++)
		img_list.Set(i, 0, i);
	img_list.SetCursor(0);
	img_list <<= THISBACK(SetImage);
	img_list.NoHeader();
	
	net_edit.SetData(t);
	net_ctrl.Add(net_edit.HSizePos().VSizePos(0,30));
	net_ctrl.Add(reload_btn.HSizePos().BottomPos(0,30));
	reload_btn.SetLabel("Reload Network");
	reload_btn <<= THISBACK(Reload);
	
	Add(img_ctrl.SizePos());
	img_ctrl.SetSession(ses);
	
	slider_ctrl.Add(lbl_slider.TopPos(0,30).HSizePos(4,4));
	slider_ctrl.Add(slider.TopPos(30,30).HSizePos(4,4));
	slider_ctrl.Add(rate_info.VSizePos(30).HSizePos(4,4));
	lbl_slider.SetLabel("Learning rate: 0.01");
	rate_info.SetLabel("The learning rate should probably be decreased over time (slide left)\nto let the network better overfit the training data.\nIt's nice to not have to worry about overfitting.");
	slider.MinMax(1, 1000);
	slider.SetData(1000);
	slider <<= THISBACK(RefreshLearningRate);
	
	ses.WhenStepInterval << THISBACK(StepInterval);
	
	PostCallback(THISBACK(SetImage));
	
	SetTimeCallback(-15, THISBACK(Refresher));
}

void RegressionPainter::DockInit() {
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
	DockLeft(Dockable(img_list, "Image List").SizeHint(Size(128, 320)));
	DockBottom(Dockable(status, "Status").SizeHint(Size(320, 200)));
	DockBottom(Dockable(slider_ctrl, "Learning rate").SizeHint(Size(320, 200)));
}

void RegressionPainter::RefreshStatus() {
	String s;
	s << "Loss: " << ses.GetLossAverage() << "\n";
	s << "Step: " << ses.GetStepCount();
	status.SetLabel(s);
}

void RegressionPainter::RefreshLearningRate() {
	double rate = (int)slider.GetData() * 0.00001;
	if (!ses.GetTrainer()) return;
	ses.GetTrainer()->SetLearningRate(rate);
	
	lbl_slider.SetLabel("Learning rate: " + DblStr(rate));
}

void RegressionPainter::Refresher() {
	RefreshStatus();
}

void RegressionPainter::Reload() {
	ses.StopTraining();
	
	String net_str = net_edit.GetData();
	
	bool success = ses.MakeLayers(net_str);
	
	if (success) {
		ses.StartTraining();
	}
}

void RegressionPainter::SetImage() {
	ses.StopTraining();
	
	int cursor = img_list.GetCursor();
	if (cursor == -1) return;
	
	Image img;
	switch (cursor) {
		case 0: img = RegressionPainterImg::cat;			break;
		case 1: img = RegressionPainterImg::battery;		break;
		case 2: img = RegressionPainterImg::chess;			break;
		case 3: img = RegressionPainterImg::chip;			break;
		case 4: img = RegressionPainterImg::dora;			break;
		case 5: img = RegressionPainterImg::earth;			break;
		case 6: img = RegressionPainterImg::esher;			break;
		case 7: img = RegressionPainterImg::fox;			break;
		case 8: img = RegressionPainterImg::fractal;		break;
		case 9: img = RegressionPainterImg::gradient;		break;
		case 10: img = RegressionPainterImg::jitendra;		break;
		case 11: img = RegressionPainterImg::pencils;		break;
		case 12: img = RegressionPainterImg::rainforest;	break;
		case 13: img = RegressionPainterImg::reddit;		break;
		case 14: img = RegressionPainterImg::rubiks;		break;
		case 15: img = RegressionPainterImg::starry;		break;
		case 16: img = RegressionPainterImg::tesla;			break;
		case 17: img = RegressionPainterImg::twitter;		break;
		case 18: img = RegressionPainterImg::usa;			break;
		default: return;
	}
	
	Size sz = img.GetSize();
	if (sz.cx * sz.cy <= 0) return;
	
	int count = sz.cx * sz.cy;
	SessionData& d = ses.Data();
	d.BeginDataResult(3, count, 2);
	
	const RGBA* it = img.Begin();
	
	int i = 0;
	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			VolumeDataBase& in_data		= d.Get(i);
			VolumeDataBase& out_data	= d.GetResult(i);
			in_data.Set(0, (double)x / sz.cx - 0.5);
			in_data.Set(1, (double)y / sz.cy - 0.5);
			out_data.Set(0, it->r / 255.0);
			out_data.Set(1, it->g / 255.0);
			out_data.Set(2, it->b / 255.0);
			it++;
			i++;
		}
	}
	
	d.EndData();
	
	Reload();
	
	slider.SetData(1000);
	lbl_slider.SetLabel("Learning rate: 0.01");
	RefreshLearningRate();
	
	img_ctrl.SetSource(img);
	
}

void RegressionPainter::StepInterval(int i) {
	if (i > 1000000) return;
	double d = min(1000.0, max(1.0, (1000000 - i) / 1000000.0 * 1000.0));
	PostCallback(THISBACK1(SetSlider, (int)d));
}
