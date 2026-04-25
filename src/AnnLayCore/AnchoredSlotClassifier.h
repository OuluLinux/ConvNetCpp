#ifndef _AnnotationEditor_AnchoredSlotClassifier_h_
#define _AnnotationEditor_AnchoredSlotClassifier_h_

#include "AnnLay.h"
#include "AnnMdl.h"
#include <ConvNet/ConvNet.h>

NAMESPACE_UPP

typedef Event<String, int, double, double> SlotTrainProgressEvent;

enum TrainingSourceMode {
	TRAIN_SOURCE_CROPS = 0,
	TRAIN_SOURCE_ANNPRJ_BOOL
};

struct TrainingGroupResult : Moveable<TrainingGroupResult> {
	String         group_key;
	bool           ok = false;
	int            epochs_done = 0;
	Vector<double> loss_history;
	Vector<double> val_acc_history;
	String         message;
};

struct TrainingJobRequest : Moveable<TrainingJobRequest> {
	TrainingSourceMode source_mode = TRAIN_SOURCE_CROPS;
	Vector<String>     group_keys;
	String             crops_dir;
	String             annprj_path;
	String             images_dir;
	int                max_epochs = 50;
	bool               balance_by_slot = true;
	int                slot_cap = 0;
	bool               ignore_auto_stop = false;
	bool               debug = false;
};

struct TrainingJobResult : Moveable<TrainingJobResult> {
	bool                       ok = false;
	Vector<TrainingGroupResult> groups;
};

class AnchoredSlotClassifier {
public:
	void TrainAll(AnnLay& lay, const String& annprj_path,
	              const String& images_dir,
	              const String& card_strip_path = String(),
	              AnnMdl* mdl = nullptr);

	void TrainSlot(AnnLay& lay, const String& slot_id,
	               const String& annprj_path,
	               const String& images_dir,
	               const String& card_strip_path = String(),
	               AnnMdl* mdl = nullptr);

	int max_epochs = 100;
	SlotTrainProgressEvent WhenProgress;

	static void BuildBoolDataset(const AnnLaySlot& slot,
	                           const AnnLay& lay,
	                           const String& annprj_path,
	                           const String& images_dir,
	                           Vector<Vector<double>>& true_samples,
	                           Vector<Vector<double>>& false_samples);

	static void BuildLabelDataset(const AnnLaySlot& slot,
	                            const AnnLay& lay,
	                            const String& annprj_path,
	                            const String& images_dir,
	                            const String& card_strip_path,
	                            Vector<Vector<double>>& samples,
	                            Vector<int>& labels);

public:

	static Image CropSlot(const Image& img, const AnnLayAnchor& anchor,
	                      double bbox_expand, Size crop_size);

	static Vector<double> ImageToSample(const Image& crop, bool grayscale, Size crop_size, bool equalize = true);

	void TrainSession(ConvNet::Session& ses,
	                  const String& net_json,
	                  int n_classes,
	                  int sample_len,
	                  int w, int h, int depth,
	                  const Vector<Vector<double>>& samples,
	                  const Vector<int>& labels,
	                  const String& slot_id,
	                  const Vector<String>* class_names = nullptr);

	void EmbedWeights(const String& slot_id, AnnMdl& mdl, ConvNet::Session& ses, const String& net_json);

public:
	// Headless training from a crops directory (subdirs = class labels).
	// slot_key: exact slot id, or a shared group key from GetSlotGroups().
	// ses: optional session to reuse (GUI can pass its own so TrainingGraph stays live).
	// Pass nullptr to use a local session (CLI headless mode).
	static VectorMap<String, Vector<String>> GetSlotGroups(const AnnLay& lay);
	static String ResolveCanonicalGroupKey(const AnnLay& lay, const String& group_key);
	static String GetGroupRole(const AnnLay& lay, const String& group_key);
	static String GetSlotGroupDisplayName(const AnnLay& lay, const String& group_key);
	static String BoolSlotGroupKey(const String& slot_id, const AnnLay* lay = nullptr);
	static double GetSlotRotation(const String& slot_id, const AnnLay* lay = nullptr);
	static bool MatchSlotForBool(const String& target_slot, const String& candidate_slot);
	static Image EqualizeHistogram(const Image& img);
	static Image LinearContrastStretching(const Image& img);
	static bool TrainBoolGroup(AnnLay& lay,
	                           const String& group_key,
	                           const String& annprj_path,
	                           const String& images_dir,
	                           int max_epochs,
	                           Event<String,int,double,double> WhenProgress,
	                           ConvNet::Session* ses = nullptr,
	                           AnnMdl* mdl = nullptr,
	                           bool balance_by_slot = false,
	                           int slot_cap = 0,
	                           bool ignore_auto_stop = false,
	                           bool debug = false);
	static bool TrainFromCropsDir(AnnLay& lay,
	                              const String& crops_dir,
	                              const String& slot_key,
	                              int max_epochs,
	                              Event<String,int,double,double> WhenProgress,
	                              ConvNet::Session* ses = nullptr,
	                              AnnMdl* mdl = nullptr,
	                              bool balance_by_slot = false,
	                              int slot_cap = 0,
	                              bool ignore_auto_stop = false,
	                              bool debug = false);
	static bool RunTrainingJob(AnnLay& lay,
	                           const TrainingJobRequest& request,
	                           TrainingJobResult& result,
	                           Event<String,int,double,double> WhenProgress = Event<String,int,double,double>(),
	                           ConvNet::Session* ses = nullptr,
	                           AnnMdl* mdl = nullptr);
};

// Keep only near-white pixels (luminance >= white_thresh); everything else → black.
Image HighLuminanceThresholdBinarization(const Image& src, double white_thresh = ANNLAY_HIGH_LUMINANCE_THRESHOLD);

END_UPP_NAMESPACE

#endif
