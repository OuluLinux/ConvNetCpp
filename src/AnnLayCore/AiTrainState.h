#ifndef _AnnLayCore_AiTrainState_h_
#define _AnnLayCore_AiTrainState_h_

#include <Core/Core.h>

NAMESPACE_UPP

struct AiTrainState {
	int            version = 1;
	String         group_key;
	Vector<String> slot_ids;
	int            epochs_completed = 0;
	int            target_epochs = 0;
	String         timestamp;
	Vector<double> loss_history;
	Vector<double> val_acc_history;
	double         last_precision = 0;
	double         last_recall = 0;
	double         last_f1 = 0;

	void Jsonize(JsonIO& jio);

	bool Load(const String& annlay_path);
	bool Save(const String& annlay_path) const;
	bool Exists(const String& annlay_path) const;
	static String SidecarPath(const String& annlay_path);
	static String GroupSidecarPath(const String& annlay_path, const String& group_key);
	static bool LoadGroup(const String& annlay_path, const String& group_key, AiTrainState& out);
};

END_UPP_NAMESPACE

#endif
