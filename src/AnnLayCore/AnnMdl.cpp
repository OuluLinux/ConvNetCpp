#include "AnnMdl.h"

NAMESPACE_UPP

String AnnMdl::PathFromAnnlay(const String& annlay_path) {
	return ForceExt(annlay_path, ".annmdl");
}

String AnnMdl::DefaultSessionStoreDirFromAnnmdl(const String& annmdl_path) {
	String base_dir = GetFileDirectory(annmdl_path);
	if(base_dir.IsEmpty())
		return String();
	return NormalizePath(AppendFileName(base_dir, "data/annmdl"));
}

void AnnMdl::SetSessionStoreDir(const String& dir) {
	String d = TrimBoth(dir);
	if(d.IsEmpty())
		session_store_dir_.Clear();
	else
		session_store_dir_ = NormalizePath(d);
}

bool AnnMdl::Load(const String& annlay_path, bool load_session_data) {
	return LoadPath(PathFromAnnlay(annlay_path), load_session_data);
}

bool AnnMdl::LoadPath(const String& path, bool load_session_data) {
	entries.Clear();
	annmdl_path_ = NormalizePath(path);
	session_store_dir_ = DefaultSessionStoreDirFromAnnmdl(annmdl_path_);

	if(!FileExists(path))
		return false;

	FileIn in(path);
	if(!in.IsOpen())
		return false;

	uint32 magic = 0;
	uint32 version = 0;
	uint32 n_slots = 0;
	in % magic % version % n_slots;
	if(in.IsError())
		return false;
	if(magic != kMagic)
		return false;
	if(version != kVersionV1 && version != kVersionV2 && version != kVersionV3)
		return false;

	entries.SetCount(n_slots);
	for(uint32 i = 0; i < n_slots; i++) {
		AnnMdlEntry& e = entries[i];
		e.session_data.Clear();
		e.session_ref.Clear();
		e.net_ref.Clear();
		e.traindata_ref.Clear();
		e.session_offset = -1;
		e.session_inline = false;

		if(version == kVersionV1) {
			in % e.slot_id % e.net_str;
			if(in.IsError()) {
				entries.Clear();
				return false;
			}

			// Record stream position for lazy inline-session loading.
			e.session_offset = in.GetPos();
			e.session_inline = true;

			String inline_blob;
			in % inline_blob;
			if(in.IsError()) {
				entries.Clear();
				return false;
			}
			if(load_session_data)
				e.session_data = inline_blob;
		}
		else if(version == kVersionV2) {
			in % e.slot_id % e.net_str % e.session_ref;
			if(in.IsError()) {
				entries.Clear();
				return false;
			}
			if(load_session_data && !e.session_ref.IsEmpty()) {
				String blob;
				if(LoadSessionData(e, blob))
					e.session_data = blob;
			}
		}
		else { // kVersionV3
			in % e.slot_id % e.net_str % e.net_ref % e.traindata_ref;
			if(in.IsError()) {
				entries.Clear();
				return false;
			}
		}
	}
	return true;
}

String AnnMdl::ResolveSessionRefPath(const String& ref) const {
	String r = TrimBoth(ref);
	if(r.IsEmpty())
		return String();
	if(IsFullPath(r))
		return NormalizePath(r);
	String base_dir = GetFileDirectory(annmdl_path_);
	if(base_dir.IsEmpty())
		return NormalizePath(r);
	return NormalizePath(AppendFileName(base_dir, r));
}

String AnnMdl::MakeSessionRef(const String& annmdl_path, const String& blob_path) const {
	String annmdl_dir = NormalizePath(GetFileDirectory(annmdl_path));
	String blob_abs = NormalizePath(blob_path);
	if(annmdl_dir.IsEmpty())
		return blob_abs;
	String prefix = annmdl_dir;
	if(!prefix.EndsWith("/"))
		prefix << "/";
	if(blob_abs.StartsWith(prefix))
		return blob_abs.Mid(prefix.GetCount());
	return blob_abs;
}

bool AnnMdl::StoreSessionBlob(const String& annmdl_path, const String& data, String& out_ref) const {
	out_ref.Clear();
	if(data.IsEmpty())
		return true;

	String store_dir = session_store_dir_;
	if(store_dir.IsEmpty())
		store_dir = DefaultSessionStoreDirFromAnnmdl(annmdl_path);
	if(store_dir.IsEmpty())
		return false;
	store_dir = NormalizePath(store_dir);

	if(!RealizeDirectory(store_dir))
		return false;

	String raw = SHA256StringS(data);
	String sha;
	for(int i = 0; i < raw.GetCount(); i++) {
		char c = raw[i];
		if((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))
			sha.Cat((char)ToLower((byte)c));
	}
	if(sha.IsEmpty())
		sha = Format("%08x", abs((int)GetHashValue(data)));
	String blob_name = sha + ".sess";
	String blob_path = NormalizePath(AppendFileName(store_dir, blob_name));
	if(!FileExists(blob_path)) {
		if(!SaveFile(blob_path, data))
			return false;
	}

	out_ref = MakeSessionRef(annmdl_path, blob_path);
	return !out_ref.IsEmpty();
}

bool AnnMdl::StoreBlobExternal(const String& annmdl_path, const String& data, const String& ext, String& out_ref) const {
	out_ref.Clear();
	if(data.IsEmpty())
		return true;

	String store_dir = session_store_dir_;
	if(store_dir.IsEmpty())
		store_dir = DefaultSessionStoreDirFromAnnmdl(annmdl_path);
	if(store_dir.IsEmpty())
		return false;
	store_dir = NormalizePath(store_dir);

	if(!RealizeDirectory(store_dir))
		return false;

	String raw = SHA256StringS(data);
	String sha;
	for(int i = 0; i < raw.GetCount(); i++) {
		char c = raw[i];
		if((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))
			sha.Cat((char)ToLower((byte)c));
	}
	if(sha.IsEmpty())
		sha = Format("%08x", abs((int)GetHashValue(data)));
	String blob_name = sha + ext;
	String blob_path = NormalizePath(AppendFileName(store_dir, blob_name));
	if(!FileExists(blob_path)) {
		if(!SaveFile(blob_path, data))
			return false;
	}

	out_ref = MakeSessionRef(annmdl_path, blob_path);
	return !out_ref.IsEmpty();
}

bool AnnMdl::LoadSessionData(const AnnMdlEntry& e, String& out) const {
	out.Clear();

	if(!e.session_data.IsEmpty()) {
		out = e.session_data;
		return true;
	}

	if(e.session_inline && e.session_offset >= 0 && !annmdl_path_.IsEmpty()) {
		FileIn in(annmdl_path_);
		if(!in.IsOpen())
			return false;
		in.Seek(e.session_offset);
		if(in.IsError())
			return false;
		String inline_blob;
		in % inline_blob;
		if(in.IsError())
			return false;
		out = inline_blob;
		return true;
	}

	if(!e.session_ref.IsEmpty()) {
		String blob_path = ResolveSessionRefPath(e.session_ref);
		if(blob_path.IsEmpty() || !FileExists(blob_path))
			return false;
		out = LoadFile(blob_path);
		return !IsNull(out);
	}

	return false;
}

bool AnnMdl::LoadSessionData(const String& slot_id, String& out) const {
	const AnnMdlEntry* e = FindEntry(slot_id);
	if(!e)
		return false;
	return LoadSessionData(*e, out);
}

bool AnnMdl::LoadWeightsData(const AnnMdlEntry& e, String& out) const {
	out.Clear();
	if(e.net_ref.IsEmpty())
		return false; // V3 only; callers should check and use LoadSessionData for V1/V2
	String blob_path = ResolveSessionRefPath(e.net_ref);
	if(blob_path.IsEmpty() || !FileExists(blob_path))
		return false;
	out = LoadFile(blob_path);
	return !IsNull(out);
}

bool AnnMdl::LoadWeightsData(const String& slot_id, String& out) const {
	const AnnMdlEntry* e = FindEntry(slot_id);
	if(!e)
		return false;
	return LoadWeightsData(*e, out);
}

bool AnnMdl::LoadTrainData(const AnnMdlEntry& e, String& out) const {
	out.Clear();
	if(!e.traindata_ref.IsEmpty()) {
		String blob_path = ResolveSessionRefPath(e.traindata_ref);
		if(blob_path.IsEmpty() || !FileExists(blob_path))
			return false;
		out = LoadFile(blob_path);
		return !IsNull(out);
	}
	return false;
}

bool AnnMdl::LoadTrainData(const String& slot_id, String& out) const {
	const AnnMdlEntry* e = FindEntry(slot_id);
	if(!e)
		return false;
	return LoadTrainData(*e, out);
}

bool AnnMdl::SavePath(const String& path) const {
	String target_path = NormalizePath(path);

	// Prepare refs before opening output (critical for in-place rewrite of V1/V2).
	Vector<String> slot_ids;
	Vector<String> net_jsons;
	Vector<String> net_refs;
	Vector<String> traindata_refs;
	int n = entries.GetCount();
	slot_ids.SetCount(n);
	net_jsons.SetCount(n);
	net_refs.SetCount(n);
	traindata_refs.SetCount(n);

	for(int i = 0; i < n; i++) {
		const AnnMdlEntry& e = entries[i];
		slot_ids[i]  = e.slot_id;
		net_jsons[i] = e.net_str;

		// --- weights blob ---
		// Preserve existing external refs for untouched entries.
		// Only rewrite when fresh in-memory blob data is provided.
		if(!e.net_data.IsEmpty()) {
			if(!StoreBlobExternal(target_path, e.net_data, ".net", net_refs[i]))
				return false;
		}
		else if(!e.net_ref.IsEmpty()) {
			net_refs[i] = e.net_ref;
		}

		// --- training-data blob ---
		if(!e.traindata_data.IsEmpty()) {
			if(!StoreBlobExternal(target_path, e.traindata_data, ".tdat", traindata_refs[i]))
				return false;
		}
		else if(!e.traindata_ref.IsEmpty()) {
			traindata_refs[i] = e.traindata_ref;
		}
	}

	FileOut out(target_path);
	if(!out.IsOpen())
		return false;

	uint32 n_slots = (uint32)n;
	uint32 magic = kMagic;
	uint32 version = kVersionV3;
	out % magic % version % n_slots;

	for(int i = 0; i < n; i++) {
		out % slot_ids[i] % net_jsons[i] % net_refs[i] % traindata_refs[i];
		if(out.IsError())
			return false;
	}
	return !out.IsError();
}


bool AnnMdl::Save(const String& annlay_path) const {
	return SavePath(PathFromAnnlay(annlay_path));
}

const AnnMdlEntry* AnnMdl::FindEntry(const String& slot_id) const {
	for(int i = 0; i < entries.GetCount(); i++)
		if(entries[i].slot_id == slot_id)
			return &entries[i];
	return nullptr;
}

AnnMdlEntry& AnnMdl::GetOrAdd(const String& slot_id) {
	for(int i = 0; i < entries.GetCount(); i++)
		if(entries[i].slot_id == slot_id)
			return entries[i];
	AnnMdlEntry& e = entries.Add();
	e.slot_id = slot_id;
	e.session_inline = false;
	e.session_offset = -1;
	return e;
}

bool AnnMdl::LoadIntoSession(const AnnMdl& mdl, const String& head_id,
                             ConvNet::Session& ses) {
	const AnnMdlEntry* entry = mdl.FindEntry(head_id);
	if(!entry || entry->net_str.IsEmpty())
		return false;
	ASSERT(!entry->net_str.IsEmpty());
	if(!ses.MakeLayers(entry->net_str))
		return false;
	if(!entry->net_ref.IsEmpty()) {
		String weights_blob;
		if(!mdl.LoadWeightsData(*entry, weights_blob) || weights_blob.IsEmpty())
			return false;
		StringStream ss(weights_blob);
		ses.SerializeWeights(ss);
	}
	else {
		String session_blob;
		if(!mdl.LoadSessionData(*entry, session_blob) || session_blob.IsEmpty())
			return false;
		StringStream ss(session_blob);
		ses.Serialize(ss);
	}
	return true;
}

END_UPP_NAMESPACE
