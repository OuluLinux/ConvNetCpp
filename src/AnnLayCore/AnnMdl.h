#ifndef _AnnotationEditor_AnnMdl_h_
#define _AnnotationEditor_AnnMdl_h_

#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

NAMESPACE_UPP

struct AnnMdlEntry : Moveable<AnnMdlEntry> {
	String slot_id;
	String net_str;
	String session_data;    // legacy: full session blob (V1/V2), in-memory only
	String session_ref;     // legacy: external blob ref (V2)
	String net_data;        // V3: weights blob, in-memory (set before Save)
	String net_ref;         // V3: weights blob external ref
	String traindata_data;  // V3: training-data blob, in-memory (set before Save)
	String traindata_ref;   // V3: training-data blob external ref
	int64  session_offset = -1; // legacy v1 inline string position in .annmdl
	bool   session_inline = false;
};

class AnnMdl {
public:
	static String PathFromAnnlay(const String& annlay_path);
	static String DefaultSessionStoreDirFromAnnmdl(const String& annmdl_path);

	bool LoadPath(const String& path, bool load_session_data = false);
	bool Load(const String& annlay_path, bool load_session_data = false);
	bool SavePath(const String& path) const;
	bool Save(const String& annlay_path) const;

	void SetSessionStoreDir(const String& dir);
	String GetSessionStoreDir() const { return session_store_dir_; }

	// Legacy (V1/V2): load the combined session blob
	bool LoadSessionData(const String& slot_id, String& out) const;
	bool LoadSessionData(const AnnMdlEntry& e, String& out) const;

	// V3: load weights-only blob (net + trainer, no training data)
	bool LoadWeightsData(const String& slot_id, String& out) const;
	bool LoadWeightsData(const AnnMdlEntry& e, String& out) const;

	// V3: load training-data blob (SessionData only)
	bool LoadTrainData(const String& slot_id, String& out) const;
	bool LoadTrainData(const AnnMdlEntry& e, String& out) const;

	const AnnMdlEntry* FindEntry(const String& slot_id) const;
	AnnMdlEntry& GetOrAdd(const String& slot_id);

	// Load entry into session: MakeLayers + V3 weights or V1/V2 session blob.
	// Returns false if entry not found, MakeLayers fails, or blob is missing.
	static bool LoadIntoSession(const AnnMdl& mdl, const String& head_id,
	                            ConvNet::Session& ses);

	Array<AnnMdlEntry> entries;

private:
	static const uint32 kMagic = 0x414D444C; // 'AMDL'
	static const uint32 kVersionV1 = 1;
	static const uint32 kVersionV2 = 2;
	static const uint32 kVersionV3 = 3;

	String annmdl_path_;
	String session_store_dir_;

	String ResolveSessionRefPath(const String& ref) const;
	String MakeSessionRef(const String& annmdl_path, const String& blob_path) const;
	bool   StoreSessionBlob(const String& annmdl_path, const String& data, String& out_ref) const;
	bool   StoreBlobExternal(const String& annmdl_path, const String& data, const String& ext, String& out_ref) const;
};

END_UPP_NAMESPACE

#endif
