#ifndef _OCR_OCRVis_h_
#define _OCR_OCRVis_h_

#include <CtrlLib/CtrlLib.h>
#include <OCR/OCRTypes.h>

namespace Upp {

namespace OCR {

// Real-time Loss and Accuracy Graphs
class TrainingGraphCtrl : public Ctrl {
public:
	typedef TrainingGraphCtrl CLASSNAME;
	TrainingGraphCtrl();

	void AddDataPoint(double loss, double accuracy);
	void Clear();

protected:
	virtual void Paint(Draw& w);

private:
	struct DataPoint {
		double loss;
		double accuracy;
	};

	Vector<DataPoint> data;
	double max_loss;
	double min_loss;
	
	Point DataToScreen(int i, double val, double min_v, double max_v);
	void DrawAxes(Draw& w);
	void DrawLegend(Draw& w);
};

// Confusion Matrix Visualization
class ConfusionMatrixCtrl : public Ctrl {
public:
	typedef ConfusionMatrixCtrl CLASSNAME;
	ConfusionMatrixCtrl();

	void SetMatrix(const Vector<Vector<int>>& matrix);
	void SetClassNames(const Vector<String>& names);
	void Clear();

protected:
	virtual void Paint(Draw& w);
	virtual void MouseMove(Point p, dword keyflags);

private:
	Vector<Vector<int>> matrix;
	Vector<String> class_names;
	Point hover_cell;

	Color GetCellColor(int value, int max_value, bool diagonal);
};

// Sample Predictions Grid
class SamplePredictionsCtrl : public Ctrl {
public:
	typedef SamplePredictionsCtrl CLASSNAME;
	SamplePredictionsCtrl();

	struct SampleResult : Moveable<SampleResult> {
		Image image;
		int true_class;
		int pred_class;
		double confidence;
		bool correct;
	};

	void SetSamples(const Vector<SampleResult>& samples);
	void Clear();

protected:
	virtual void Paint(Draw& w);

private:
	Vector<SampleResult> samples;
	int cell_size;
	
	void DrawSample(Draw& w, const SampleResult& sample, Rect r);
};

// Text Splitter Visualization
class SplitterVisCtrl : public Ctrl {
public:
	typedef SplitterVisCtrl CLASSNAME;
	SplitterVisCtrl();

	void SetImage(const Image& img);
	void SetPredictedBoundaries(const Vector<int>& bounds);
	void SetGroundTruth(const Vector<int>& bounds);
	void Clear();

protected:
	virtual void Paint(Draw& w);

private:
	Image image;
	Vector<int> predicted;
	Vector<int> ground_truth;
};

// Class Distribution Bar Chart
class ClassDistributionCtrl : public Ctrl {
public:
	typedef ClassDistributionCtrl CLASSNAME;
	ClassDistributionCtrl();

	void SetDistribution(const Vector<int>& counts);
	void Clear();

protected:
	virtual void Paint(Draw& w);

private:
	Vector<int> counts;
};

} // namespace OCR

} // namespace Upp

#endif
