#include "AiTrainState.h"
#include "GroupRegistry.h"

NAMESPACE_UPP

namespace {

bool ContainsKey(const Vector<String>& keys, const String& key) {
	for(const String& k : keys)
		if(k == key)
			return true;
	return false;
}

double SanitizeFinite(double v, double fallback = 0.0) {
	return IsFin(v) ? v : fallback;
}

void SanitizeHistory(Vector<double>& v) {
	for(int i = 0; i < v.GetCount(); i++)
		v[i] = SanitizeFinite(v[i], 0.0);
}

void SanitizeState(AiTrainState& s) {
	SanitizeHistory(s.loss_history);
	SanitizeHistory(s.val_acc_history);
	s.last_precision = SanitizeFinite(s.last_precision, 0.0);
	s.last_recall = SanitizeFinite(s.last_recall, 0.0);
	s.last_f1 = SanitizeFinite(s.last_f1, 0.0);
}

}

static bool LoadTrainStateFromPath(const String& path, AiTrainState& out) {
	if(!FileExists(path))
		return false;
	try {
		String data = LoadFile(path);
		if(data.IsEmpty())
			return false;
		if(!LoadFromJson(out, data))
			return false;
		SanitizeState(out);
		return true;
	}
	catch(...) {
		return false;
	}
}

void AiTrainState::Jsonize(JsonIO& jio) {
	jio("version", version)
	   ("group_key", group_key)
	   ("slot_ids", slot_ids)
	   ("epochs_completed", epochs_completed)
	   ("target_epochs", target_epochs)
	   ("timestamp", timestamp)
	   ("loss_history", loss_history)
	   ("val_acc_history", val_acc_history)
	   ("last_precision", last_precision)
	   ("last_recall", last_recall)
	   ("last_f1", last_f1);
}

String AiTrainState::SidecarPath(const String& annlay_path) {
	return annlay_path + ".trainstate";
}

String AiTrainState::GroupSidecarPath(const String& annlay_path, const String& group_key) {
	return annlay_path + "." + group_key + ".trainstate";
}

bool AiTrainState::Exists(const String& annlay_path) const {
	return FileExists(SidecarPath(annlay_path));
}

bool AiTrainState::Load(const String& annlay_path) {
	return LoadTrainStateFromPath(SidecarPath(annlay_path), *this);
}

bool AiTrainState::LoadGroup(const String& annlay_path, const String& group_key, AiTrainState& out) {
	Vector<String> try_keys;
	if(!group_key.IsEmpty())
		try_keys.Add(group_key);
	AnnLay lay;
	if(lay.Load(annlay_path)) {
		GroupRegistry reg;
		reg.Build(lay);
		String canonical = reg.Resolve(group_key);
		if(!canonical.IsEmpty() && !ContainsKey(try_keys, canonical))
			try_keys.Add(canonical);
		Vector<String> aliases = reg.LegacyAliases(canonical.IsEmpty() ? group_key : canonical);
		for(const String& a : aliases)
			if(!a.IsEmpty() && !ContainsKey(try_keys, a))
				try_keys.Add(a);
	}
	for(const String& k : try_keys) {
		if(LoadTrainStateFromPath(GroupSidecarPath(annlay_path, k), out))
			return true;
	}
	// Fallback to legacy path if group-specific doesn't exist
	return LoadTrainStateFromPath(SidecarPath(annlay_path), out);
}

bool AiTrainState::Save(const String& annlay_path) const {
	try {
		AiTrainState out;
		out.version = version;
		out.group_key = group_key;
		out.slot_ids = clone(slot_ids);
		out.epochs_completed = epochs_completed;
		out.target_epochs = target_epochs;
		out.timestamp = timestamp;
		out.loss_history = clone(loss_history);
		out.val_acc_history = clone(val_acc_history);
		out.last_precision = last_precision;
		out.last_recall = last_recall;
		out.last_f1 = last_f1;
		SanitizeState(out);
		String path = group_key.IsEmpty() ? SidecarPath(annlay_path) : GroupSidecarPath(annlay_path, group_key);
		return SaveFile(path, StoreAsJson(out, true));
	}
	catch(...) {
		return false;
	}
}

END_UPP_NAMESPACE
