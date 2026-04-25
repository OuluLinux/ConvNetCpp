#ifndef _OCRTraining_OCRTraining_h_
#define _OCRTraining_OCRTraining_h_

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <Docking/Docking.h>
#include <OCR/OCRDatasetProject.h>
#include <OCR/OCRTypes.h>
#include <OCR/OCRModel.h>
#include <OCR/NetworkBuilder.h>
#include <OCR/TextSplitter.h>
#include <OCRCtrl/OCRCtrl.h>
#include <OCRCtrl/OCRVis.h>

namespace Upp {

using namespace ConvNet;

// Stub for LayerCtrl until dependency issue is resolved
class LayerCtrl : public ParentCtrl {
public:
	void SetSession(Session& s) {}
	void UpdateLayers() {}
	Callback1<int> WhenViewLayer;
};

class OCRTraining : public DockWindow {
public:
	typedef OCRTraining CLASSNAME;
	OCRTraining();
	~OCRTraining();

	virtual void DockInit();

	void MainMenu(Bar& bar);
	void FileMenu(Bar& bar);
	void TrainingMenu(Bar& bar);

	// Mode
	enum Mode { MODE_CLASSIFIER, MODE_SPLITTER };
	void SetMode(int m);

	// Data loading
	void LoadProject(const String& path);
	void LoadDatasetProject();
	void SyncDatasetToSession();
	void SyncClassifier();
	void SyncSplitter();

	// Training
	void StartTraining();
	void StopTraining();
	void ResetNetwork();
	void ReloadNetwork();
	void ExportModel();
	void RunBatchTest();

	// Visualization
	void ViewLayer(int i);
	void Refresher();
	void HelperThread();
	void ApplyVisuals();

	void SetEmbedded(bool b = true);

private:
	bool docking_initialized = false;
	
	// Helper Thread Data
	Mutex helper_mutex;
	bool helper_running = false;
	int vis_iter;
	double vis_loss, vis_acc;
	int vis_samples;
	Vector<Vector<int>> vis_m;
	Vector<OCR::SamplePredictionsCtrl::SampleResult> vis_sr;
	
	// Session & Data
	Session session;
	OCR::OCRDatasetProject dataset;
	int mode;
	bool running, stopped;

	// GUI Components - Network Editor
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	Button reload_btn;

	// GUI Components - Layers
	LayerCtrl lctrl;
	
	// GUI Components - Visualization
	OCR::TrainingGraphCtrl graph;
	OCR::ConfusionMatrixCtrl confusion;
	OCR::SamplePredictionsCtrl samples;
	OCR::SplitterVisCtrl splitter_vis;
	OCR::ClassDistributionCtrl dist_ctrl;

	// GUI Components - Control Panel
	ParentCtrl control_panel;
	DropList mode_select;
	Button btn_load_dataset;
	Button btn_start_training;
	Button btn_stop_training;
	Button btn_reset_network;
	Button btn_export;
	Button btn_test;
	Label status_label;

	void UpdateStatus();
};

} // namespace Upp

#endif
