#include "NetworkOptimization.h"

#include "NetworkOptimization.brc"
#include <plugin/bz2/bz2.h>

#define IMAGECLASS NetworkOptimizationImg
#define IMAGEFILE <NetworkOptimization/NetworkOptimization.iml>
#include <Draw/iml_source.h>

NetworkOptimization::NetworkOptimization() {
	Title("Network optimization demo");
	Icon(NetworkOptimizationImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	
	trainer_running = false;
	trainer_stopped = true;
	
	iter = 0;

	data_ctrl.Add(import_split.TopPos(0,30).HSizePos());
	data_ctrl.Add(data_doc.VSizePos(30,30).HSizePos(4,4));
	data_ctrl.Add(lbl_testfac.BottomPos(0,30).HSizePos(100,200));
	data_ctrl.Add(testfac.BottomPos(0,30).LeftPos(4,100-8));
	data_ctrl.Add(import.BottomPos(0,30).RightPos(0,200));
	
	best_ctrl.Add(best_net.HSizePos().VSizePos(0,30));
	best_ctrl.Add(export_best.BottomPos(0,30).HCenterPos(200,0));
	export_best.SetLabel("Export best");
	export_best		<<= THISBACK(RefreshBestNetwork);
	
	test_ctrl.Add(test_model_count.TopPos(0,30).LeftPos(0,50));
	test_ctrl.Add(lbl_test_model_count.TopPos(0,30).HSizePos(50,0));
	test_ctrl.Add(test_data.HSizePos().VSizePos(30,30));
	test_ctrl.Add(eval_test.BottomPos(0,30).HCenterPos(200,0));
	test_model_count.SetData(10);
	lbl_test_model_count.SetLabel("Number of best models to average in the ensemble network");
	eval_test.SetLabel("Evaluate Test accuracy");
	eval_test		<<= THISBACK(TestEvaluation0);
	
	lbl_testfac.SetLabel("\% to send to to test set");
	testfac.SetData(20);
	testfac.MinMax(1, 99);
	
	cross_ctrl.Add(idx					.LeftPos(0,50).TopPos(0,30));
	cross_ctrl.Add(lbl_idx				.HSizePos(50,0).TopPos(0,30));
	cross_ctrl.Add(train_perc			.LeftPos(0,50).TopPos(30,30));
	cross_ctrl.Add(lbl_train_perc		.HSizePos(50,0).TopPos(30,30));
	cross_ctrl.Add(folds_per_cand		.LeftPos(0,50).TopPos(60,30));
	cross_ctrl.Add(lbl_folds_per_cand	.HSizePos(50,0).TopPos(60,30));
	cross_ctrl.Add(cands_per_batch		.LeftPos(0,50).TopPos(90,30));
	cross_ctrl.Add(lbl_cands_per_batch	.HSizePos(50,0).TopPos(90,30));
	cross_ctrl.Add(epochs_per_fold		.LeftPos(0,50).TopPos(120,30));
	cross_ctrl.Add(lbl_epochs_per_fold	.HSizePos(50,0).TopPos(120,30));
	cross_ctrl.Add(min_neur				.LeftPos(0,50).TopPos(150,30));
	cross_ctrl.Add(lbl_min_neur			.HSizePos(50,0).TopPos(150,30));
	cross_ctrl.Add(max_neur				.LeftPos(0,50).TopPos(180,30));
	cross_ctrl.Add(lbl_max_neur			.HSizePos(50,0).TopPos(180,30));
	cross_ctrl.Add(restart				.HCenterPos(200,0).TopPos(210,30));
	restart.SetLabel("Restart");
	restart <<= THISBACK(StartTrainer);
	
	lbl_idx.SetLabel(" Index of column to classify as target. (e.g. 0 = first column, -1 = last column) ");
	lbl_train_perc.SetLabel(" Percent of data to use for training (rest will be validation) ");
	lbl_folds_per_cand.SetLabel(" Number of data folds to evaluate per candidate ");
	lbl_cands_per_batch.SetLabel(" Number of candidates in a batch, to evaluate in parallel ");
	lbl_epochs_per_fold.SetLabel(" Number of epochs to make over each fold ");
	lbl_min_neur.SetLabel(" Min number of neurons in each layer");
	lbl_max_neur.SetLabel(" Max number of neurons in each layer");
	
	idx						.SetData(-1);
	train_perc				.SetData(70);
	folds_per_cand			.SetData(1);
	cands_per_batch			.SetData(50);
	epochs_per_fold			.SetData(20);
	min_neur				.SetData(5);
	max_neur				.SetData(30);
	
	import_split << import_iris << import_careval << import_yeast;
	import_split.Horz();
	
	import_iris		.SetLabel("Fill Iris data");
	import_careval	.SetLabel("Fill Car Eval. data");
	import_yeast	.SetLabel("Fill Yeast data");
	import			.SetLabel("Import");
	
	import_careval	<<= THISBACK1(FillData, 0);
	import_iris		<<= THISBACK1(FillData, 1);
	import_yeast	<<= THISBACK1(FillData, 2);
	import			<<= THISBACK(ImportTrainData);
	
	log.SetFont(Monospace(13));
	
	status_ctrl.Add(status.HSizePos(4,4).VSizePos(4,4));
	
	FillData(1);
	
	Add(plot.SizePos());
	
	mnet.WhenFinishFold << THISBACK(PostClearPlotter);
	mnet.WhenFinishBatch << THISBACK(PostRefreshBestNetwork);
	
	
	ImportTrainData();
}

NetworkOptimization::~NetworkOptimization() {
	StopTrainer();
}

void NetworkOptimization::DockInit() {
	DockLeft(Dockable(data_ctrl, "Import data").SizeHint(Size(320, 200)));
	DockLeft(Dockable(cross_ctrl, "Cross-Validation").SizeHint(Size(320, 200)));
	DockLeft(Dockable(log, "Log").SizeHint(Size(320, 200)));
	DockBottom(Dockable(status_ctrl, "Status").SizeHint(Size(320, 100)));
	AutoHide(DOCK_LEFT, Dockable(best_ctrl, "View Best Network").SizeHint(Size(640, 320)));
	AutoHide(DOCK_LEFT, Dockable(test_ctrl, "Evaluate on Test set").SizeHint(Size(640, 320)));
}

void NetworkOptimization::StartTrainer() {
	StopTrainer();
	
	ClearPlotter();
	
	mnet.SetTrainingRatio( (int)train_perc.GetData() / 100.0 );
	mnet.SetFoldsCount( folds_per_cand.GetData() );
	mnet.SetCandidateCount( cands_per_batch.GetData() );
	mnet.SetEpochCount( epochs_per_fold );
	mnet.SetNeuronRange( min_neur.GetData(), max_neur.GetData() );
	
	ContinueTrainer();
}

void NetworkOptimization::ContinueTrainer() {
	trainer_running = true;
	trainer_stopped = false;
	Thread::Start(THISBACK(Runner));
}

void NetworkOptimization::RefreshStatus() {
	if (mnet.GetEvaluatedCandidateCount() > 0 || mnet.GetSessionCount() > 0) {
		Session& best_cand = mnet.GetEvaluatedCandidateCount() > 0 ?
			mnet.GetEvaluatedCandidate(0) :
			mnet.GetSessions()[0];
		String s;
		s << "Validation accuracy of best model so far, overall: " << best_cand.GetValidationAccuracyAverage() << "\n";
		s << "Net layer definitions:\n" << best_cand.GetNetwork().ToString() << "\n";
		s << "Trainer definition:\n" << best_cand.GetTrainer()->ToString() << "\n";
		status.SetLabel(s);
	}
}

void NetworkOptimization::RefreshBestNetwork() {
	if (mnet.GetEvaluatedCandidateCount() > 0 || mnet.GetSessionCount() > 0) {
		Session& best_cand = mnet.GetEvaluatedCandidateCount() > 0 ?
			mnet.GetEvaluatedCandidate(0) :
			mnet.GetSessions()[0];
		String net;
		best_cand.StoreOriginalJSON(net);
		best_net.SetData(net);
	}
}

void NetworkOptimization::FillData(int src) {
	if (src < 0 || src >= data_txt_count) return;
	
	MemReadStream mem_in(data_txt[src], data_txt_length[src]);
	String txt = BZ2Decompress(mem_in);
	data_doc.SetData(txt);
	
}

ColumnData NetworkOptimization::GuessColumn(Vector<Vector<String> >& data, int c) {
	bool numeric = true;
	Index<String> uniques;
	for(int i = 0; i < data.GetCount(); i++) {
		const String& v = data[i][c];
		for(int j = 0; j < v.GetCount(); j++) {
			char chr = v[j];
			if (IsDigit(chr) || chr == '.' || chr == '-' || chr == '+') continue;
			numeric = false;
			break;
		}
		if (uniques.Find(v) == -1)
			uniques.Add(v);
	}
	
	if (!numeric) {
		// if we have a non-numeric we will map it through uniques to an index
		return ColumnData(numeric, uniques.GetCount(), uniques);
	} else {
		return ColumnData(numeric, uniques.GetCount());
	}
}

void NetworkOptimization::ImportData(Vector<Vector<String> >& arr, Data& data) {
	// find number of datapoints
	N = arr.GetCount();
	
	AddLog("Found " + IntStr(N) + " data points");
	if (N == 0) {
		AddLog("No data points found?");
		return;
	}
	
	// find dimensionality and enforce consistency
	D = arr[0].GetCount();
	for (int i = 0; i < N; i++) {
		int d = arr[i].GetCount();
		if (d != D) {
			AddLog("Data dimension not constant: line " + IntStr(i) + " has " + IntStr(d) + " entries.");
			return;
		}
	}
	AddLog("data dimensionality is " + IntStr(D-1));
	
	// go through columns of data and figure out what they are
	data.Clear();
	data.arr <<= arr;
	for(int i = 0; i < D; i++) {
		ColumnData& res = data.colstats.Add(GuessColumn(arr, i));
		if (D > 20 && i > 3 && i < D-3) {
			if(i==4) {
				AddLog("..."); // suppress output for too many columns
			}
		} else {
			AddLog("Column " + IntStr(i) + " looks " + (res.numeric ? "numeric" : "NOT numeric") + " and has " + IntStr(res.num) + " unique elements");
		}
	}
}

void NetworkOptimization::MakeDataset(Vector<Vector<String> >& arr, Vector<ColumnData>& colstats, SessionData& ds) {
	int labelix = idx.GetData();
	if (labelix < 0)
		labelix = D + labelix; // -1 should turn to D-1
	
	int cls_count = colstats[labelix].uniques.GetCount();
	
	ds.ClearData();
	ds.BeginData(cls_count, N, colstats.GetCount());
	
	for (int i = 0; i < N; i++) {
		Vector<String>& arri = arr[i];
		
		// create the input datapoint Vol()
		int pos = 0;
		for (int j = 0; j < D; j++) {
			if (j == labelix) continue; // skip!
			
			if (colstats[j].numeric) {
				ds.SetData(i, pos, StrDbl(arri[j]));
			} else {
				Index<String>& u = colstats[j].uniques;
				int ix = u.Find(arri[j]); // turn into 1ofk encoding
				for (int q = 0; q < u.GetCount(); q++) {
					if (q == ix) {
						ds.SetData(i, pos, 1.0);
					}
					else {
						ds.SetData(i, pos, 0.0);
					}
				}
			}
			
			pos++;
		}
		
		int L;
		if (colstats[labelix].numeric) {
			L = StrDbl(arri[labelix]); // regression
		} else {
			L = colstats[labelix].uniques.Find(arri[labelix]); // classification
			if (L == -1) {
				AddLog("whoa label not found! CRITICAL ERROR, very fishy.");
			}
		}
		
		ds.SetLabel(i, L);
		
	}
	
	ds.EndData();
	
	mnet.SampleFolds();
	mnet.SampleCandidates();
}

void NetworkOptimization::TestEvaluation0() {
	TestEvaluation(mnet);
}

void NetworkOptimization::TestEvaluation(MagicNet& net) {
	StopTrainer();
	
	SessionData& data = mnet.GetSessionData(0);
	
	// set options for magic net
	int ensemble_size = test_model_count.GetData();
	
	// read in the data in the text field
	ImportTestData();
	
	// use magic net to predict
	int n = data.GetDataCount();
	double acc = 0.0;
	
	Volume v(data.GetDataWidth(), data.GetDataHeight(), data.GetDataDepth(), 0);
	
	for (int i = 0; i < n; i++) {
		
		v.SetData(data.Get(i));
		
		int yhat = net.PredictSoftLabel(v);
		
		if (yhat == -1) {
			return;
		}
		int l = data.GetLabel(i);
		acc += (yhat == l ? 1 : 0); // 0-1 loss
		AddLog("test example " + IntStr(i) + ": predicting " + IntStr(yhat) + ", ground truth is " + IntStr(l));
	}
	acc /= n;
	
	// report accuracy
	AddLog("Test set accuracy: " + DblStr(acc));
	
	
	ContinueTrainer();
}

void NetworkOptimization::Runner() {
	while (trainer_running && !Thread::IsShutdownThreads()) {
		Step();
	}
	trainer_stopped = true;
}

void NetworkOptimization::RefreshPlotter() {
	plot.Sync();
	plot.Refresh();
	RefreshStatus();
}

void NetworkOptimization::Step() {
	iter++;
	
	mnet.Step();
	
	if(iter % 300 == 0) {
		
		Vector<double> vals;
		mnet.EvaluateValueErrors(vals);
		
		
		plot_lock.Enter();
		
		int mnet_iter = mnet.GetIteration();
		int session_count = mnet.GetSessionCount();
		for(int i = 0; i < session_count; i++) {
			const Session& ses = mnet.GetSession(i);
			double d = vals[i];
			plot.data[i].AddXY(mnet_iter, d);
		}
		
		plot.SetLimits(plot.data[0][0].x, mnet_iter, 0, 1);
		plot.SetModify();
		
		plot_lock.Leave();
		
		
		PostCallback(THISBACK(RefreshPlotter));
	}
}

void NetworkOptimization::ImportTrainData() {
	StopTrainer();
	
	String csv_txt = data_doc.GetData();
	Vector<String> lines = Split(csv_txt, "\n");
	Vector<Vector<String> > arr;
	Vector<Vector<String> > arr_train;
	Vector<Vector<String> > arr_test;
	
	arr.SetCount(lines.GetCount());
	for(int i = 0; i < lines.GetCount(); i++) {
		arr[i] = Split(lines[i], ",");
	}
	
	int test_ratio = testfac.GetData();
	
	if (test_ratio != 0) {
		 
		// send some lines to test set
		int test_lines_num = floor(arr.GetCount() * test_ratio / 100.0);
		Vector<int> rp;
		RandomPermutation(arr.GetCount(), rp);
		
		for (int i = 0; i < rp.GetCount(); i++) {
			if (i < test_lines_num) {
				arr_test.Add() <<= arr[rp[i]];
			} else {
				arr_train.Add() <<= arr[rp[i]];
			}
		}
	} else {
		arr_train <<= arr;
	}
	
	
	// enter test lines to test box
	String test_data;
	for(int i = 0; i < arr_test.GetCount(); i++) {
		String line = Join(arr_test[i], ",");
		test_data << line << "\n";
	}
	this->test_data.SetData(test_data);
	
	
	AddLog("Sent " + IntStr(arr_test.GetCount()) + " data to test, keeping " + IntStr(arr_train.GetCount()) + " for train.");
	
	ImportData(arr_train, train_import_data);
	MakeDataset(train_import_data.arr, train_import_data.colstats, mnet.GetSessionData(0));
	
	StartTrainer();
}

void NetworkOptimization::ImportTestData() {
	String csv_txt = data_doc.GetData();
	Vector<String> lines = Split(csv_txt, "\n");
	Vector<Vector<String> > arr;
	
	arr.SetCount(lines.GetCount());
	for(int i = 0; i < lines.GetCount(); i++) {
		arr[i] = Split(lines[i], ",");
	}
	
	Data import_data;
	ImportData(arr, import_data);
	
	// note important that we use colstats of train data!
	MakeDataset(import_data.arr, train_import_data.colstats, mnet.GetSessionData(0));
	
}

void NetworkOptimization::AddLog(const String& line) {
	LOG(line);
	log.Insert(log.GetLength(), line + "\n");
	log.SetCursor(log.GetLength());
}

void NetworkOptimization::ClearPlotter() {
	
	plot_lock.Enter();
	
	// PlotCtrl has zooming prompt bugs, so we create new one when resetting
	
	plot.data.SetCount( mnet.GetSessionCount());
	for(int i = 0; i < mnet.GetSessionCount(); i++) {
		plot.data[i].SetCount(0);
		plot.data[i]
			.SetTitle("Model " + IntStr(i))
			.SetThickness(1.0)
			.SetColor(Rainbow((double)i / mnet.GetSessionCount()));
		plot.data[i].AddXY(0,0);
	}
	plot.SetLimits(-1, 1, -1, 1);
	plot.SetModify();
	
	plot_lock.Leave();
	
}
