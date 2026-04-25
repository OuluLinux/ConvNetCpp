#include "AiTrainFile.h"

NAMESPACE_UPP

bool AiTrainFile::Load(const String& path) {
	try {
		String json = LoadFile(path);
		if(json.IsEmpty())
			return false;

		Value root = ParseJSON(json);
		if(IsNull(root) || !IsValueMap(root))
			return false;

		int file_version = root["version"];
		if(file_version <= 0)
			file_version = 1;
		if(file_version > 1)
			return false;

		version = file_version;
		String type_v = root["type"];
		type = type_v.IsEmpty() ? String("slot_classifier_train") : type_v;

		entries.Clear();
		Value rows = root["entries"];
		if(IsValueArray(rows)) {
			for(int i = 0; i < rows.GetCount(); i++) {
				const Value& row = rows[i];
				if(!IsValueMap(row))
					continue;

				AiTrainEntry& e = entries.Add();
				e.id = row["id"];
				e.annlay = row["annlay"];
				e.crops_dir = row["crops_dir"];

				String slot_key = row["slot"];
				if(!slot_key.IsEmpty())
					e.slot = slot_key;

				int epochs_v = row["epochs"];
				if(epochs_v > 0)
					e.epochs = epochs_v;
			}
		}
		return true;
	}
	catch(...) {
		return false;
	}
}

bool AiTrainFile::Save(const String& path) const {
	try {
		ValueMap root;
		root.Add("version", 1);
		root.Add("type", type);

		ValueArray rows;
		for(const AiTrainEntry& e : entries) {
			ValueMap row;
			row.Add("id", e.id);
			row.Add("annlay", e.annlay);
			row.Add("crops_dir", e.crops_dir);
			row.Add("slot", e.slot);
			row.Add("epochs", e.epochs > 0 ? e.epochs : 1);
			rows.Add(row);
		}
		root.Add("entries", rows);

		return SaveFile(path, StoreAsJson(root, true));
	}
	catch(...) {
		return false;
	}
}

END_UPP_NAMESPACE
