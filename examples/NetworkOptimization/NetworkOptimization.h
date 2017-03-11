#ifndef _NetworkOptimization_NetworkOptimization_h
#define _NetworkOptimization_NetworkOptimization_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS NetworkOptimizationImg
#define IMAGEFILE <NetworkOptimization/NetworkOptimization.iml>
#include <Draw/iml_header.h>


struct ColumnData : Moveable<ColumnData> {
	ColumnData() {}
	ColumnData(const ColumnData& src) {*this = src;}
	ColumnData(bool numeric, int num) : numeric(numeric), num(num) {}
	ColumnData(bool numeric, int num, const Index<String>& uniques) : numeric(numeric), num(num) {this->uniques <<= uniques;}
	ColumnData& operator=(const ColumnData& src) {
		numeric = src.numeric;
		num = src.num;
		uniques <<= src.uniques;
		return *this;
	}
	
	bool numeric;
	int num;
	Index<String> uniques;
};

struct Data {
	void Clear() {arr.Clear(); colstats.Clear();}
	Vector<Vector<String> > arr;
	Vector<ColumnData> colstats;
};

class NetworkOptimization : public DockWindow {
	
	MagicNet mnet;
	
	// Best network
	ParentCtrl best_ctrl;
	DocEdit best_net;
	Button export_best;
	
	// Test data
	EditIntSpin test_model_count;
	Label lbl_test_model_count;
	ParentCtrl test_ctrl;
	DocEdit test_data;
	Button eval_test;
	
	// Log
	DocEdit log;
	
	// Import training data
	ParentCtrl data_ctrl;
	DocEdit data_doc;
	Splitter import_split;
	Button import_iris, import_careval, import_yeast;
	Button import;
	Label lbl_testfac;
	EditIntSpin testfac;
	
	// Cross-Validation
	ParentCtrl cross_ctrl;
	EditIntSpin idx, train_perc, folds_per_cand, cands_per_batch, epochs_per_fold, min_neur, max_neur;
	Label lbl_idx, lbl_train_perc, lbl_folds_per_cand, lbl_cands_per_batch, lbl_epochs_per_fold, lbl_min_neur, lbl_max_neur;
	Button restart;
	
	// Status
	ParentCtrl status_ctrl;
	Label status;
	
	// Main view
	MetaSessionGraph graph;
	Mutex plot_lock;
	
	
	Data train_import_data;
	
	String t;
	int N, D;
	int iter;
	bool trainer_running, trainer_stopped;
	
public:
	typedef NetworkOptimization CLASSNAME;
	NetworkOptimization();
	~NetworkOptimization();
	
	virtual void DockInit();
	
	void ClearLog() {log.Clear();}
	void AddLog(const String& line);
	void TestEvaluation(MagicNet& net);
	void TestEvaluation0();
	void MakeDataset(Vector<Vector<String> >& arr, Vector<ColumnData>& colstats, SessionData& ds);
	ColumnData GuessColumn(Vector<Vector<String> >& data, int c);
	void FillData(int src);
	void ImportData(Vector<Vector<String> >& arr, Data& data);
	void Start();
	void Runner();
	void StartTrainer();
	void ContinueTrainer();
	void StopTrainer() {trainer_running = false; while (!trainer_stopped) Sleep(100);}
	void RefreshStatus();
	void RefreshBestNetwork();
	void PostRefreshBestNetwork() {PostCallback(THISBACK(RefreshBestNetwork));}
	void ImportTrainData();
	void ImportTestData();
	
};

#endif
