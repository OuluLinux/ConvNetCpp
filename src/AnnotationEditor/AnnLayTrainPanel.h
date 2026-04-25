#ifndef _AnnotationEditor_AnnLayTrainPanel_h_
#define _AnnotationEditor_AnnLayTrainPanel_h_

#include <AnnLayCore/AnnLay.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <CtrlLib/CtrlLib.h>

NAMESPACE_UPP

class AnnLayTrainPanel : public ParentCtrl {
public:
	typedef AnnLayTrainPanel CLASSNAME;

	AnnLayTrainPanel();
	~AnnLayTrainPanel() override;
	void ConfigureForStartup(const String& crops_dir, const String& annlay_file);

private:
	EditString         edit_crops_dir;
	Button             btn_browse_crops;
	EditString         edit_annlay_path;
	Button             btn_browse_annlay;
	DropList           drop_slot;
	EditInt            edit_epochs;
	Button             btn_start;
	Button             btn_stop;
	Button             btn_save;
	Label              lbl_status;
	Splitter           h_split;
	ConvNet::TrainingGraph      graph;
	ConvNet::SessionConvLayers  layer_view;
	ConvNet::ImagePrediction    pred_view;

	ConvNet::Session   ses;
	AnnLay             lay;
	String             annlay_path;
	Thread             train_thread;
	Mutex              lock;

	Vector<Vector<double>> loaded_samples;
	Vector<int>            loaded_labels;
	Vector<String>         trained_slot_ids;
	String                 trained_slot_key;
	String                 train_crops_dir;
	String                 train_net_json;

	int                target_epochs = 50;
	bool               stop_requested = false;
	bool               train_running = false;
	bool               train_done = false;
	int                trained_epochs = 0;
	double             last_loss = 0;
	double             last_val_acc = 0;
	Vector<double>     loss_history;
	Vector<double>     val_acc_history;
	String             last_message;

	void BrowseCropsDir();
	void BrowseAnnLayPath();
	bool LoadAnnLay();
	void ReloadSlots();
	Vector<String> ResolveTargetSlots(const String& slot_key) const;
	String BuildNetJson(int w, int h, int d, int cls_count) const;
	Vector<double> ImageToSample(const Image& crop, Size crop_size) const;
	bool LoadCrops();
	void StartTrain();
	void StopTrain(bool wait_thread = false);
	void TrainingMain();
	void SaveAnnLay();
	void Refresher();
	void SetStatus(const String& s);
	void SetControlsEnabled();
};

END_UPP_NAMESPACE

#endif
