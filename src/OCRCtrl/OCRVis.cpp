#include "OCRVis.h"

namespace Upp {

namespace OCR {

// --- TrainingGraphCtrl ---

TrainingGraphCtrl::TrainingGraphCtrl() {
	max_loss = 1.0;
	min_loss = 0.0;
	BackPaint();
}

void TrainingGraphCtrl::AddDataPoint(double loss, double accuracy) {
	DataPoint& p = data.Add();
	p.loss = loss;
	p.accuracy = accuracy;
	if (loss > max_loss) max_loss = loss;
	if (data.GetCount() > 1000) data.Remove(0);
	Refresh();
}

void TrainingGraphCtrl::Clear() {
	data.Clear();
	max_loss = 1.0;
	min_loss = 0.0;
	Refresh();
}

Point TrainingGraphCtrl::DataToScreen(int i, double val, double min_v, double max_v) {
	Size sz = GetSize();
	int x = 40 + i * (sz.cx - 50) / max(1, data.GetCount() - 1);
	int y = sz.cy - 30 - (int)((val - min_v) * (sz.cy - 50) / max(0.0001, max_v - min_v));
	return Point(x, y);
}

void TrainingGraphCtrl::DrawAxes(Draw& w) {
	Size sz = GetSize();
	w.DrawLine(40, 10, 40, sz.cy - 30, 1, SColorText()); // Y
	w.DrawLine(40, sz.cy - 30, sz.cx - 10, sz.cy - 30, 1, SColorText()); // X
	
	w.DrawText(5, 10, Format("%.1f", max_loss), StdFont(8));
	w.DrawText(5, sz.cy - 40, "0.0", StdFont(8));
}

void TrainingGraphCtrl::DrawLegend(Draw& w) {
	w.DrawText(50, 10, "Loss (Red)", StdFont().Bold(), Red());
	w.DrawText(150, 10, "Accuracy (Green)", StdFont().Bold(), Green());
}

void TrainingGraphCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	
	if (data.IsEmpty()) {
		w.DrawText(sz.cx / 2 - 50, sz.cy / 2, "No training data", StdFont(), SColorDisabled());
		return;
	}

	DrawAxes(w);
	DrawLegend(w);

	// Draw accuracy (green)
	for (int i = 1; i < data.GetCount(); i++) {
		w.DrawLine(DataToScreen(i-1, data[i-1].accuracy, 0, 1),
		           DataToScreen(i, data[i].accuracy, 0, 1), 2, Green());
	}

	// Draw loss (red)
	for (int i = 1; i < data.GetCount(); i++) {
		w.DrawLine(DataToScreen(i-1, data[i-1].loss, 0, max_loss),
		           DataToScreen(i, data[i].loss, 0, max_loss), 2, Red());
	}
}

// --- ConfusionMatrixCtrl ---

ConfusionMatrixCtrl::ConfusionMatrixCtrl() {
	hover_cell = Point(-1, -1);
	BackPaint();
}

void ConfusionMatrixCtrl::SetMatrix(const Vector<Vector<int>>& m) {
	matrix = clone(m);
	Refresh();
}

void ConfusionMatrixCtrl::SetClassNames(const Vector<String>& names) {
	class_names = clone(names);
	Refresh();
}

void ConfusionMatrixCtrl::Clear() {
	matrix.Clear();
	Refresh();
}

void ConfusionMatrixCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	
	if (matrix.IsEmpty()) return;
	
	int n = matrix.GetCount();
	int margin = 30;
	int cell_sz = min(sz.cx - margin, sz.cy - margin) / n;
	if (cell_sz < 1) cell_sz = 1;
	
	int max_val = 1;
	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
			max_val = max(max_val, matrix[i][j]);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			Rect r(margin + j * cell_sz, margin + i * cell_sz, margin + (j + 1) * cell_sz, margin + (i + 1) * cell_sz);
			w.DrawRect(r, GetCellColor(matrix[i][j], max_val, i == j));
			if (n < 30) DrawFrame(w, r, SColorDisabled());
			
			if (hover_cell == Point(j, i)) {
				DrawFrame(w, r, Red());
			}
		}
	}
}

Color ConfusionMatrixCtrl::GetCellColor(int value, int max_value, bool diagonal) {
	if (value == 0) return SColorPaper();
	int ratio = 255 * value / max_value;
	if (diagonal) return Color(255 - ratio, 255, 255 - ratio);
	return Color(255, 255 - ratio, 255 - ratio);
}

void ConfusionMatrixCtrl::MouseMove(Point p, dword keyflags) {
	if (matrix.IsEmpty()) return;
	int n = matrix.GetCount();
	int margin = 30;
	int cell_sz = min(GetSize().cx - margin, GetSize().cy - margin) / n;
	if (cell_sz < 1) return;
	
	Point cell((p.x - margin) / cell_sz, (p.y - margin) / cell_sz);
	if (cell.x >= 0 && cell.x < n && cell.y >= 0 && cell.y < n) {
		if (cell != hover_cell) {
			hover_cell = cell;
			Refresh();
		}
	} else if (hover_cell.x != -1) {
		hover_cell = Point(-1, -1);
		Refresh();
	}
}

// --- SamplePredictionsCtrl ---

SamplePredictionsCtrl::SamplePredictionsCtrl() {
	cell_size = 80;
	BackPaint();
}

void SamplePredictionsCtrl::SetSamples(const Vector<SampleResult>& s) {
	samples = clone(s);
	Refresh();
}

void SamplePredictionsCtrl::Clear() {
	samples.Clear();
	Refresh();
}

void SamplePredictionsCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	
	if (samples.IsEmpty()) return;
	
	int cols = sz.cx / cell_size;
	if (cols < 1) cols = 1;
	
	for (int i = 0; i < samples.GetCount(); i++) {
		int row = i / cols;
		int col = i % cols;
		Rect r(col * cell_size, row * cell_size, (col + 1) * cell_size, (row + 1) * cell_size);
		DrawSample(w, samples[i], r);
	}
}

void SamplePredictionsCtrl::DrawSample(Draw& w, const SampleResult& s, Rect r) {
	Rect img_r = r.Deflated(5);
	img_r.bottom -= 20;
	
	w.DrawImage(img_r, s.image);
	DrawFrame(w, img_r, s.correct ? Green() : Red());
	
	String txt = Format("%c->%c (%.0f%%)", ClassToChar(s.true_class), ClassToChar(s.pred_class), s.confidence * 100);
	w.DrawText(r.left + 2, r.bottom - 15, txt, StdFont(8), s.correct ? SColorText() : Red());
}

// --- SplitterVisCtrl ---

SplitterVisCtrl::SplitterVisCtrl() {
	BackPaint();
}

void SplitterVisCtrl::SetImage(const Image& img) {
	image = img;
	Refresh();
}

void SplitterVisCtrl::SetPredictedBoundaries(const Vector<int>& bounds) {
	predicted = clone(bounds);
	Refresh();
}

void SplitterVisCtrl::SetGroundTruth(const Vector<int>& bounds) {
	ground_truth = clone(bounds);
	Refresh();
}

void SplitterVisCtrl::Clear() {
	image = Image();
	predicted.Clear();
	ground_truth.Clear();
	Refresh();
}

void SplitterVisCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	
	if (image.IsEmpty()) return;
	
	Size img_sz = image.GetSize();
	double zoom = min((double)sz.cx / img_sz.cx, (double)sz.cy / img_sz.cy);
	
	int dw = (int)(img_sz.cx * zoom);
	int dh = (int)(img_sz.cy * zoom);
	
	w.DrawImage(0, 0, dw, dh, image);
	
	for (int x : ground_truth) {
		int sx = (int)(x * zoom);
		w.DrawLine(sx, 0, sx, dh, 2, LtGreen());
	}
	
	for (int x : predicted) {
		int sx = (int)(x * zoom);
		w.DrawLine(sx, 0, sx, dh, 1, LtRed());
	}
}

// --- ClassDistributionCtrl ---

ClassDistributionCtrl::ClassDistributionCtrl() {
	BackPaint();
}

void ClassDistributionCtrl::SetDistribution(const Vector<int>& c) {
	counts = clone(c);
	Refresh();
}

void ClassDistributionCtrl::Clear() {
	counts.Clear();
	Refresh();
}

void ClassDistributionCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	
	if (counts.IsEmpty()) return;
	
	int n = counts.GetCount();
	int max_val = 1;
	for(int c : counts) max_val = max(max_val, c);
	
	int bw = sz.cx / n;
	if (bw < 1) bw = 1;
	
	for (int i = 0; i < n; i++) {
		int bh = counts[i] * (sz.cy - 20) / max_val;
		Rect r(i * bw + 1, sz.cy - 20 - bh, (i + 1) * bw - 1, sz.cy - 20);
		w.DrawRect(r, Color(200, 200, 255));
		DrawFrame(w, r, SColorText());
		
		if (bw > 8) {
			String label = Format("%c", ClassToChar(i));
			w.DrawText(i * bw + (bw - 6) / 2, sz.cy - 15, label, StdFont(10));
		}
	}
}

} // namespace OCR

} // namespace Upp
