#ifndef _AnnLayCore_GroupRegistry_h_
#define _AnnLayCore_GroupRegistry_h_

#include "AnnLay.h"
#include "AnnSln.h"

NAMESPACE_UPP

struct GroupRegistryEntry : Moveable<GroupRegistryEntry> {
	String id;
	String display_name;
	String head_role;
	String preprocess;
	String export_dataset_dir;
	AnnLayMethod method = ANNLAY_IGNORED;
	Vector<String> legacy_aliases;
	Index<String> slot_ids;
};

class GroupRegistry {
public:
	void Build(const AnnLay& lay, const AnnSln* sln = nullptr);

	String Resolve(const String& any_key) const;
	const GroupRegistryEntry* Find(const String& any_key) const;

	String DisplayName(const String& any_key) const;
	String HeadRole(const String& any_key) const;
	String Preprocess(const String& any_key) const;
	String ExportDatasetDir(const String& any_key) const;
	bool   IsBoolGroup(const String& any_key) const;

	Vector<String> LegacyAliases(const String& any_key) const;
	String HeadSuffixForGroup(const String& any_key) const;

private:
	VectorMap<String, GroupRegistryEntry> entries;
	VectorMap<String, String> alias_to_id;

	GroupRegistryEntry& GetOrAdd(const String& id);
	void AddAlias(const String& alias, const String& canonical);
	void AttachSlot(const String& id, const String& slot_id);
};

END_UPP_NAMESPACE

#endif
