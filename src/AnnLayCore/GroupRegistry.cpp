#include "GroupRegistry.h"

NAMESPACE_UPP

namespace {

String TrimKey(const String& s) {
	return TrimBoth(s);
}

String NormalizeLegacyKey(const String& s) {
	String k = TrimKey(s);
	if(k == "s|h|d|c|")
		return "card_suit";
	return k;
}

bool RoleWantsLegacyAlias(const String& role) {
	return role == "presence" || role == "level" || role == "level" || role == "category" || role == "category";
}

bool ContainsValue(const Vector<String>& values, const String& value) {
	for(const String& v : values)
		if(v == value)
			return true;
	return false;
}

}

GroupRegistryEntry& GroupRegistry::GetOrAdd(const String& id) {
	String k = NormalizeLegacyKey(id);
	GroupRegistryEntry& e = entries.GetAdd(k);
	if(e.id.IsEmpty())
		e.id = k;
	return e;
}

void GroupRegistry::AddAlias(const String& alias, const String& canonical) {
	String a = NormalizeLegacyKey(alias);
	String c = NormalizeLegacyKey(canonical);
	if(a.IsEmpty() || c.IsEmpty())
		return;
	if(alias_to_id.Find(a) < 0)
		alias_to_id.Add(a, c);
}

void GroupRegistry::AttachSlot(const String& id, const String& slot_id) {
	if(id.IsEmpty() || slot_id.IsEmpty())
		return;
	GetOrAdd(id).slot_ids.FindAdd(slot_id);
}

void GroupRegistry::Build(const AnnLay& lay, const AnnSln* sln) {
	entries.Clear();
	alias_to_id.Clear();

	for(const AnnLayGroup& g : lay.groups) {
		GroupRegistryEntry& e = GetOrAdd(g.id);
		e.method = g.method;
		e.display_name = g.display_name.IsEmpty() ? e.id : g.display_name;
		e.head_role = TrimKey(g.head_role);
		e.preprocess = TrimKey(g.preprocess);
		e.export_dataset_dir = TrimKey(g.export_dataset_dir);
		e.legacy_aliases = clone(g.legacy_aliases);
		AddAlias(e.id, e.id);
		for(const String& a : e.legacy_aliases)
			AddAlias(a, e.id);
	}

	for(const AnnLaySlot& slot : lay.slots) {
		if(!slot.group.IsEmpty()) {
			AttachSlot(slot.group, slot.id);
			AddAlias(slot.group, slot.group);
		}

		for(int i = 0; i < slot.sub_groups.GetCount(); i++) {
			String role = TrimKey(slot.sub_groups.GetKey(i));
			String gid = NormalizeLegacyKey(slot.sub_groups[i]);
			if(gid.IsEmpty())
				continue;
			GroupRegistryEntry& e = GetOrAdd(gid);
			if(e.display_name.IsEmpty())
				e.display_name = gid;
			if(e.head_role.IsEmpty())
				e.head_role = role;
			AttachSlot(gid, slot.id);
			AddAlias(gid, gid);

			// sub-head ids (if provided) should resolve to the configured group id.
			String head_id = TrimKey(slot.sub_heads.Get(role, ""));
			if(!head_id.IsEmpty())
				AddAlias(head_id, gid);
		}
	}

	// Compatibility aliases kept in one place.
	for(int i = 0; i < entries.GetCount(); i++) {
		const GroupRegistryEntry& e = entries[i];
		String role = e.head_role;
		if(!RoleWantsLegacyAlias(role))
			continue;
		if(role == "presence") {
			AddAlias("card_visibility_gate", e.id);
			AddAlias("element#presence", e.id);
			AddAlias("element#presence", e.id);
		}
		if(role == "level" || role == "level") {
			AddAlias("element#level", e.id);
			AddAlias("element#level", e.id);
			AddAlias("level", e.id);
		}
		if(role == "category" || role == "category") {
			AddAlias("element#category", e.id);
			AddAlias("element#category", e.id);
			AddAlias("s|h|d|c|", e.id);
			AddAlias("card_suit", e.id);
			AddAlias("category", e.id);
		}
	}

	(void)sln; // reserved for future cv_template-group derived aliases
}

String GroupRegistry::Resolve(const String& any_key) const {
	String key = NormalizeLegacyKey(any_key);
	if(key.IsEmpty())
		return key;
	if(entries.Find(key) >= 0)
		return key;
	int q = alias_to_id.Find(key);
	if(q >= 0)
		return alias_to_id[q];
	return key;
}

const GroupRegistryEntry* GroupRegistry::Find(const String& any_key) const {
	String k = Resolve(any_key);
	int q = entries.Find(k);
	return q >= 0 ? &entries[q] : nullptr;
}

String GroupRegistry::DisplayName(const String& any_key) const {
	const GroupRegistryEntry* e = Find(any_key);
	if(!e)
		return Resolve(any_key);
	return e->display_name.IsEmpty() ? e->id : e->display_name;
}

String GroupRegistry::HeadRole(const String& any_key) const {
	const GroupRegistryEntry* e = Find(any_key);
	return e ? e->head_role : String();
}

String GroupRegistry::Preprocess(const String& any_key) const {
	const GroupRegistryEntry* e = Find(any_key);
	return e ? e->preprocess : String();
}

String GroupRegistry::ExportDatasetDir(const String& any_key) const {
	const GroupRegistryEntry* e = Find(any_key);
	if(!e)
		return Resolve(any_key);
	return e->export_dataset_dir.IsEmpty() ? e->id : e->export_dataset_dir;
}

bool GroupRegistry::IsBoolGroup(const String& any_key) const {
	const GroupRegistryEntry* e = Find(any_key);
	if(!e)
		return false;
	return e->method == ANNLAY_CLASSIFIER_BOOL;
}

Vector<String> GroupRegistry::LegacyAliases(const String& any_key) const {
	Vector<String> out;
	String canonical = Resolve(any_key);
	const GroupRegistryEntry* e = Find(canonical);
	if(!e)
		return out;
	for(const String& a : e->legacy_aliases)
		if(TrimKey(a) != canonical)
			out.Add(TrimKey(a));

	for(int i = 0; i < alias_to_id.GetCount(); i++) {
		if(alias_to_id[i] == canonical) {
			String a = alias_to_id.GetKey(i);
			if(a != canonical && !ContainsValue(out, a))
				out.Add(a);
		}
	}
	return out;
}

String GroupRegistry::HeadSuffixForGroup(const String& any_key) const {
	String role = HeadRole(any_key);
	if(!role.IsEmpty())
		return "#" + role;
	String canonical = Resolve(any_key);
	int hash = canonical.Find('#');
	if(hash >= 0)
		return canonical.Mid(hash);
	return String();
}

END_UPP_NAMESPACE
