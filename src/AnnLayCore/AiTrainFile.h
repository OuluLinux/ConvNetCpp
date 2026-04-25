#ifndef _AnnLayCore_AiTrainFile_h_
#define _AnnLayCore_AiTrainFile_h_

#include <Core/Core.h>

NAMESPACE_UPP

struct AiTrainEntry : Moveable<AiTrainEntry> {
	String id;
	String annlay;
	String crops_dir;
	String slot   = "all_cards";
	int    epochs = 50;
};

class AiTrainFile {
public:
	int               version = 1;
	String            type = "slot_classifier_train";
	Array<AiTrainEntry> entries;

	bool Load(const String& path);
	bool Save(const String& path) const;
};

END_UPP_NAMESPACE

#endif
