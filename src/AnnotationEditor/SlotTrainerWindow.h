#ifndef _AnnotationEditor_SlotTrainerWindow_h_
#define _AnnotationEditor_SlotTrainerWindow_h_

#include <AnnLayCore/AnnLay.h>
#include <AnnLayCore/AnnMdl.h>
#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNet/ConvNet.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <atomic>

NAMESPACE_UPP

class SlotTrainerWindow : public DockWindow {
public:
	typedef SlotTrainerWindow CLASSNAME;

	SlotTrainerWindow();
	~SlotTrainerWindow() override;

	void DockInit() override;
	void Configure(const String& annlay,
	               const String& crops_dir,
	               const String& slot_key = String(),
	               const String& annprj = String(),
	               const String& images_dir = String());

private:
	Splitter     v_split;
	MenuBar      menu_bar;
	ParentCtrl   settings_panel;

	Label        lbl_crops_dir;
	Label        lbl_annprj;
	Label        lbl_images_dir;
	Label        lbl_annlay;
	Label        lbl_slot;
	Label        lbl_epochs;
	EditString   edit_crops_dir;
	Button       btn_browse_crops;
	EditString   edit_annprj_path;
	Button       btn_browse_annprj;
	EditString   edit_images_dir;
	Button       btn_browse_images;
	EditString   edit_annlay_path;
	Button       btn_browse_annlay;
	DropList     drop_slot;
	EditInt      edit_epochs;
	Option       opt_auto_stop;
	Button       btn_start;
	Button       btn_stop;
	Label        lbl_status;
	Label        lbl_metrics;

	ConvNet::TrainingGraph      graph;
	ConvNet::SessionConvLayers  layer_view;
	ConvNet::ImagePrediction    pred_view;

	ConvNet::Session   ses;
	AnnLay             lay;
	String             annlay_path;
	Thread             train_thread;
	Mutex              lock;
	String             train_crops_dir;
	String             train_annprj_path;
	String             train_images_dir;
	String             train_group_key;
	AnnMdl             train_mdl;
	bool               train_auto_stop = true;
	String                 trained_slot_key;

	Vector<String>     trained_slot_ids;
	Vector<double>     loss_history;
	Vector<double>     val_acc_history;
	bool               sidecar_loaded = false;
	String             sidecar_group_key;
	Vector<String>     sidecar_slot_ids;
	Vector<double>     sidecar_loss_history;
	Vector<double>     sidecar_val_acc_history;
	int                sidecar_epochs_completed = 0;
	int                sidecar_target_epochs = 0;
	double             sidecar_last_precision = 0;
	double             sidecar_last_recall = 0;
	double             sidecar_last_f1 = 0;
	String             sidecar_timestamp;
	int                target_epochs = 50;
	std::atomic_bool   auto_stop_triggered{false};
	std::atomic_bool   stop_requested{false};
	std::atomic_bool   train_running{false};
	std::atomic_bool   closing{false};
	std::atomic<int>   continue_decision{0}; // 0=wait, 1=yes, 2=no
	bool               train_done = false;
	int                trained_epochs = 0;
	double             last_loss = 0;
	double             last_val_acc = 0;
	BoolGateMetrics    metrics;
	String             last_message;
	bool               train_is_bool = false;

	void BrowseCropsDir();
	void BrowseAnnprjPath();
	void BrowseImagesDir();
	void BrowseAnnLayPath();
	bool LoadAnnLay();
	bool IsCurrentGroupBool() const;
	void UpdateTrainingModeUI();
	void RefreshPreviewForSelection();
	void OnSlotSelectionChanged();

	void ReloadSlots();
	void RehookSession();
	Vector<String> ResolveTargetSlots(const String& slot_key) const;
	String BuildNetJson(int w, int h, int d, int cls_count) const;
	Vector<double> ImageToSample(const Image& crop, Size crop_size) const;
	bool AnySlotHasWeights(const Vector<String>& slot_ids, const String& slot_key) const;
	void RestoreSessionFromWeights(const Vector<String>& slot_ids, const String& slot_key);
	void StartTrain();
	void StopTrain(bool wait_thread = false);
	void TrainingMain();
	void ComputeBoolMetrics();
	void DoSave();
	void OnTrainingCompletedPrompt();
	void OnTrainingStoppedPrompt();
	void Refresher();
	void SetStatus(const String& s);
	void SetControlsEnabled();
	void MainMenu(Bar& bar);
};

END_UPP_NAMESPACE

#endif
