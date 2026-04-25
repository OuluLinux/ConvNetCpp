#include <Core/Core.h>
#include <AnnotationEditor/AnchoredSlotClassifier.h>

using namespace Upp;

namespace {

bool RequireValue(const Vector<String>& cmdline, int& i, String& out, const char* name) {
	if(i + 1 >= cmdline.GetCount()) {
		Cout() << "missing value for " << name << "\n";
		return false;
	}
	out = cmdline[++i];
	return true;
}

bool IsTrainableMethod(AnnLayMethod m) {
	return m == ANNLAY_CLASSIFIER_BOOL || m == ANNLAY_CLASSIFIER_LABEL;
}

int CountTrainableSlots(const AnnLay& lay) {
	int n = 0;
	for(int i = 0; i < lay.slots.GetCount(); i++)
		if(IsTrainableMethod(lay.slots[i].method))
			n++;
	return n;
}

void PrintUsage() {
	Cout() << "AnnLayTrain usage:\n"
	       << "  --annlay <path>   (required)\n"
	       << "  --annprj <path>   (required)\n"
	       << "  --images <dir>    (optional, default empty)\n"
	       << "  --epochs <n>      (optional, default 100)\n"
	       << "  --out <path>      (optional, default --annlay path)\n"
	       << "  --slot <id>       (optional, train only one slot)\n";
}

}

CONSOLE_APP_MAIN {
	String annlay_path;
	String annprj_path;
	String images_dir;
	String out_path;
	String slot_id;
	int max_epochs = 100;

	const Vector<String>& cmdline = CommandLine();
	for(int i = 0; i < cmdline.GetCount(); i++) {
		const String& a = cmdline[i];
		if(a == "--annlay") {
			if(!RequireValue(cmdline, i, annlay_path, "--annlay")) return;
		}
		else if(a == "--annprj") {
			if(!RequireValue(cmdline, i, annprj_path, "--annprj")) return;
		}
		else if(a == "--images") {
			if(!RequireValue(cmdline, i, images_dir, "--images")) return;
		}
		else if(a == "--epochs") {
			String s;
			if(!RequireValue(cmdline, i, s, "--epochs")) return;
			max_epochs = StrInt(s);
			if(max_epochs <= 0)
				max_epochs = 100;
		}
		else if(a == "--out") {
			if(!RequireValue(cmdline, i, out_path, "--out")) return;
		}
		else if(a == "--slot") {
			if(!RequireValue(cmdline, i, slot_id, "--slot")) return;
		}
		else if(a == "--help" || a == "-h") {
			PrintUsage();
			return;
		}
		else {
			Cout() << "unknown argument: " << a << "\n";
			PrintUsage();
			return;
		}
	}

	if(annlay_path.IsEmpty() || annprj_path.IsEmpty()) {
		Cout() << "--annlay and --annprj are required\n";
		PrintUsage();
		return;
	}
	if(out_path.IsEmpty())
		out_path = annlay_path;

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		Cout() << "failed to load annlay: " << annlay_path << "\n";
		return;
	}

	AnchoredSlotClassifier trainer;
	trainer.max_epochs = max_epochs;
	trainer.WhenProgress = [max_epochs](String slot_id, int epoch, double loss, double val_acc) {
		if(epoch < 0)
			epoch = 0;
		int shown_epoch = epoch;
		if(max_epochs > 0 && shown_epoch > max_epochs)
			shown_epoch = max_epochs;
		Cout() << slot_id << "  "
		       << shown_epoch << "/" << max_epochs << "  "
		       << loss << "  " << val_acc << "\n";
	};

	int trained_slots = 0;
	if(slot_id.IsEmpty()) {
		trained_slots = CountTrainableSlots(lay);
		trainer.TrainAll(lay, annprj_path, images_dir);
	}
	else {
		const AnnLaySlot* slot = lay.FindSlot(slot_id);
		if(!slot) {
			Cout() << "slot not found: " << slot_id << "\n";
			return;
		}
		if(!IsTrainableMethod(slot->method)) {
			Cout() << "slot is not trainable (method=" << AnnLayMethodToString(slot->method)
			       << "): " << slot_id << "\n";
			return;
		}
		trained_slots = 1;
		trainer.TrainSlot(lay, slot_id, annprj_path, images_dir);
	}

	if(!lay.Save(out_path)) {
		Cout() << "failed to save annlay: " << out_path << "\n";
		return;
	}

	Cout() << "slots_trained=" << trained_slots << "\n";
	Cout() << "out_annlay=" << out_path << "\n";
}
