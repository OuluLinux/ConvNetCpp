#include "AppMainCommands.h"
#include <AnnLayCore/AnnMdl.h>
#include <AnnLayCore/AnnSln.h>
#include <AnnLayCore/AnnLay.h>
#include <AnnLayCore/GroupRegistry.h>

namespace Upp {

bool RunMigrateAnnmdlSessions(const AppArguments& args) {
	String sln_path = NormalizePath(args.migrate_annmdl_sln_path);
	if(sln_path.IsEmpty() || !FileExists(sln_path)) {
		Cout() << "migrate-annmdl-sessions: .annsln file not found: " << sln_path << "\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "migrate-annmdl-sessions: failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	Index<String> annmdl_paths;
	for(int i = 0; i < sln.model_sets.GetCount(); i++) {
		String p = TrimBoth(sln.model_sets[i]);
		if(p.IsEmpty())
			continue;
		if(!IsFullPath(p))
			p = NormalizePath(AppendFileName(sln_dir, p));
		annmdl_paths.FindAdd(p);
	}

	String annlay_path = TrimBoth(sln.annlay);
	if(!annlay_path.IsEmpty()) {
		if(!IsFullPath(annlay_path))
			annlay_path = NormalizePath(AppendFileName(sln_dir, annlay_path));
		annmdl_paths.FindAdd(AnnMdl::PathFromAnnlay(annlay_path));
	}

	if(annmdl_paths.IsEmpty()) {
		Cout() << "migrate-annmdl-sessions: no model files found from .annsln\n";
		return false;
	}

	String out_dir = TrimBoth(args.migrate_annmdl_out_dir);
	if(!out_dir.IsEmpty() && !IsFullPath(out_dir))
		out_dir = NormalizePath(AppendFileName(sln_dir, out_dir));

	int migrated = 0;
	for(int i = 0; i < annmdl_paths.GetCount(); i++) {
		String annmdl_path = annmdl_paths[i];
		if(!FileExists(annmdl_path)) {
			Cout() << "migrate-annmdl-sessions: skip missing file: " << annmdl_path << "\n";
			continue;
		}

		AnnMdl mdl;
		if(!mdl.LoadPath(annmdl_path, false)) {
			Cout() << "migrate-annmdl-sessions: failed to load: " << annmdl_path << "\n";
			return false;
		}

		String backup_path = annmdl_path + ".bak";
		if(!FileExists(backup_path)) {
			if(!FileCopy(annmdl_path, backup_path)) {
				Cout() << "migrate-annmdl-sessions: failed to create backup: " << backup_path << "\n";
				return false;
			}
		}

		if(!out_dir.IsEmpty())
			mdl.SetSessionStoreDir(out_dir);

		if(!mdl.SavePath(annmdl_path)) {
			Cout() << "migrate-annmdl-sessions: failed to save migrated file: " << annmdl_path << "\n";
			return false;
		}

		String store_dir = mdl.GetSessionStoreDir();
		if(store_dir.IsEmpty())
			store_dir = AnnMdl::DefaultSessionStoreDirFromAnnmdl(annmdl_path);

		Cout() << "migrate-annmdl-sessions: migrated " << annmdl_path
		       << " -> session store: " << store_dir << "\n";
		migrated++;
	}

	Cout() << "migrate-annmdl-sessions: done files_migrated=" << migrated << "\n";
	return migrated > 0;
}

bool RunMigrateAnnmdlV3(const AppArguments& args) {
	String sln_path = NormalizePath(args.migrate_v3_sln_path);
	if(sln_path.IsEmpty() || !FileExists(sln_path)) {
		Cout() << "migrate-annmdl-v3: .annsln file not found: " << sln_path << "\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "migrate-annmdl-v3: failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	Index<String> annmdl_paths;
	for(int i = 0; i < sln.model_sets.GetCount(); i++) {
		String p = TrimBoth(sln.model_sets[i]);
		if(p.IsEmpty())
			continue;
		if(!IsFullPath(p))
			p = NormalizePath(AppendFileName(sln_dir, p));
		annmdl_paths.FindAdd(p);
	}
	String annlay_path = TrimBoth(sln.annlay);
	if(!annlay_path.IsEmpty()) {
		if(!IsFullPath(annlay_path))
			annlay_path = NormalizePath(AppendFileName(sln_dir, annlay_path));
		annmdl_paths.FindAdd(AnnMdl::PathFromAnnlay(annlay_path));
	}

	if(annmdl_paths.IsEmpty()) {
		Cout() << "migrate-annmdl-v3: no model files found from .annsln\n";
		return false;
	}

	String out_dir = TrimBoth(args.migrate_v3_out_dir);
	if(!out_dir.IsEmpty() && !IsFullPath(out_dir))
		out_dir = NormalizePath(AppendFileName(sln_dir, out_dir));

	int migrated = 0;
	for(int i = 0; i < annmdl_paths.GetCount(); i++) {
		String annmdl_path = annmdl_paths[i];
		if(!FileExists(annmdl_path)) {
			Cout() << "migrate-annmdl-v3: skip missing file: " << annmdl_path << "\n";
			continue;
		}

		// Load with full session data so we can split it
		AnnMdl mdl;
		mdl.LoadPath(annmdl_path, true);

		// Check if already V3 (all entries have net_ref set)
		bool already_v3 = mdl.entries.GetCount() > 0;
		for(int ei = 0; ei < mdl.entries.GetCount(); ei++) {
			if(mdl.entries[ei].net_ref.IsEmpty()) {
				already_v3 = false;
				break;
			}
		}
		if(already_v3) {
			Cout() << "migrate-annmdl-v3: already V3, skipping: " << annmdl_path << "\n";
			continue;
		}

		// Split each entry's legacy blob into weights + traindata
		for(int ei = 0; ei < mdl.entries.GetCount(); ei++) {
			AnnMdlEntry& e = mdl.entries[ei];
			if(e.session_data.IsEmpty() && e.session_ref.IsEmpty() && !e.session_inline)
				continue;

			String blob;
			if(e.session_data.IsEmpty())
				mdl.LoadSessionData(e, blob);
			else
				blob = e.session_data;

			if(blob.IsEmpty()) {
				Cout() << "migrate-annmdl-v3: no session data for slot '" << e.slot_id << "' in " << annmdl_path << "\n";
				continue;
			}

			ConvNet::Session ses;
			if(!e.net_str.IsEmpty())
				ses.MakeLayers(e.net_str);
			{
				StringStream ss(blob);
				ses.Serialize(ss);
			}

			{
				StringStream ss;
				ses.SerializeWeights(ss);
				e.net_data = ss.GetResult();
			}
			{
				StringStream ss;
				ses.SerializeTrainData(ss);
				e.traindata_data = ss.GetResult();
			}

			// Clear legacy fields so SavePath writes V3 only
			e.session_data.Clear();
			e.session_ref.Clear();
			e.session_inline = false;
			e.session_offset = -1;
		}

		String backup_path = annmdl_path + ".bak";
		if(!FileExists(backup_path)) {
			if(!FileCopy(annmdl_path, backup_path)) {
				Cout() << "migrate-annmdl-v3: failed to create backup: " << backup_path << "\n";
				return false;
			}
		}

		if(!out_dir.IsEmpty())
			mdl.SetSessionStoreDir(out_dir);

		if(!mdl.SavePath(annmdl_path)) {
			Cout() << "migrate-annmdl-v3: failed to save migrated file: " << annmdl_path << "\n";
			return false;
		}

		String store_dir = mdl.GetSessionStoreDir();
		if(store_dir.IsEmpty())
			store_dir = AnnMdl::DefaultSessionStoreDirFromAnnmdl(annmdl_path);

		Cout() << "migrate-annmdl-v3: migrated " << annmdl_path
		       << " -> blob store: " << store_dir << "\n";
		migrated++;
	}

	Cout() << "migrate-annmdl-v3: done files_migrated=" << migrated << "\n";
	return migrated > 0;
}

bool RunMergeModes(const AppArguments& args) {
	return RunMergeAnnMdl(args.merge_src_path, args.merge_dst_path);
}

bool RunMergeAnnMdl(const String& src_path, const String& dst_path) {
	AnnMdl src, dst;
	// Load with session data to ensure we can move/merge correctly
	if(!src.LoadPath(src_path, true)) {
		Cout() << "merge-annmdl: failed to load src: " << src_path << "\n";
		return false;
	}
	if(!dst.LoadPath(dst_path, true)) {
		Cout() << "merge-annmdl: failed to load dst: " << dst_path << "\n";
		return false;
	}

	for(int i = 0; i < src.entries.GetCount(); i++) {
		const AnnMdlEntry& e_src = src.entries[i];
		AnnMdlEntry& e_dst = dst.GetOrAdd(e_src.slot_id);
		e_dst.net_str = e_src.net_str;
		e_dst.session_data = e_src.session_data;
		e_dst.session_ref = e_src.session_ref;
		e_dst.session_offset = e_src.session_offset;
		e_dst.session_inline = e_src.session_inline;
	}

	if(!dst.SavePath(dst_path)) {
		Cout() << "merge-annmdl: failed to save merged file to: " << dst_path << "\n";
		return false;
	}

	Cout() << "merge-annmdl: success merged " << src.entries.GetCount() << " heads from " << src_path << " to " << dst_path << "\n";
	return true;
}

// Returns the slot_id of the entry in mdl whose net_data/session_data is non-empty and whose
// slot_id ends with "#<suffix>", preferring the first such entry found.
// Returns empty string if none found.
static String FindPerSlotDonor(const AnnMdl& mdl, const String& suffix) {
	String sfx = "#" + suffix;
	for(int i = 0; i < mdl.entries.GetCount(); i++) {
		const AnnMdlEntry& e = mdl.entries[i];
		if(!e.slot_id.EndsWith(sfx))
			continue;
		if(!e.net_data.IsEmpty() || !e.net_ref.IsEmpty() ||
		   !e.session_data.IsEmpty() || !e.session_ref.IsEmpty() || e.session_offset >= 0)
			return e.slot_id;
	}
	return String();
}

// Copy blob references/data from src entry to dst entry (weights + traindata paths).
// If dst is V3, copies net_ref/net_data. If src is legacy, copies session_data/session_ref.
static void CopyEntryBlobs(AnnMdlEntry& dst, const AnnMdlEntry& src) {
	dst.net_str      = src.net_str;
	dst.net_data     = src.net_data;
	dst.net_ref      = src.net_ref;
	dst.traindata_data = src.traindata_data;
	dst.traindata_ref  = src.traindata_ref;
	// Legacy fields (V1/V2 compatibility)
	dst.session_data   = src.session_data;
	dst.session_ref    = src.session_ref;
	dst.session_offset = src.session_offset;
	dst.session_inline = src.session_inline;
}

bool RunNormalizeCompositeHeads(const AppArguments& args) {
	bool dry_run = args.normalize_composite_dry_run;
	String label = dry_run ? "normalize-composite-heads (dry-run)" : "normalize-composite-heads";

	String sln_path = NormalizePath(args.normalize_composite_sln_path);
	if(sln_path.IsEmpty() || !FileExists(sln_path)) {
		Cout() << label << ": .annsln file not found: " << sln_path << "\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << label << ": failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);

	String annlay_path = TrimBoth(sln.annlay);
	if(!annlay_path.IsEmpty() && !IsFullPath(annlay_path))
		annlay_path = NormalizePath(AppendFileName(sln_dir, annlay_path));
	if(annlay_path.IsEmpty() || !FileExists(annlay_path)) {
		Cout() << label << ": annlay not found: " << annlay_path << "\n";
		return false;
	}

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		Cout() << label << ": failed to load annlay: " << annlay_path << "\n";
		return false;
	}

	// Collect all canonical group IDs needed for composite element sub-roles.
	// For each composite element slot, read sub_groups: suffix → canonical_group_id.
	// We also need per-role canonical group IDs and their legacy aliases to find donors.
	// Structure: role_suffix -> canonical_group_id (deduped across all slots)
	VectorMap<String, String> role_to_canonical; // e.g. "presence" -> "card_visibility_gate"
	for(int si = 0; si < lay.slots.GetCount(); si++) {
		const AnnLaySlot& slot = lay.slots[si];
		if(slot.composite_type == ANNLAY_COMPOSITE_NONE)
			continue;
		
		for(int ri = 0; ri < slot.sub_groups.GetCount(); ri++) {
			String role = slot.sub_groups.GetKey(ri);
			String gid  = slot.sub_groups[ri];
			if(role.IsEmpty() || gid.IsEmpty())
				continue;
			if(role_to_canonical.Find(role) < 0)
				role_to_canonical.Add(role, gid);
		}
	}

	if(role_to_canonical.IsEmpty()) {
		Cout() << label << ": no composite element slots with sub_groups found in annlay\n";
		return true;
	}

	Cout() << label << ": composite roles found:\n";
	for(int ri = 0; ri < role_to_canonical.GetCount(); ri++)
		Cout() << "  " << role_to_canonical.GetKey(ri) << " -> " << role_to_canonical[ri] << "\n";

	// Build group registry to find legacy aliases
	GroupRegistry reg;
	reg.Build(lay);

	// Collect all legacy aliases per canonical group (reverse map: canonical -> aliases)
	// We use reg.LegacyAliases() to enumerate them.

	// Process each annmdl referenced by the sln
	Index<String> annmdl_paths;
	for(int i = 0; i < sln.model_sets.GetCount(); i++) {
		String p = TrimBoth(sln.model_sets[i]);
		if(p.IsEmpty()) continue;
		if(!IsFullPath(p))
			p = NormalizePath(AppendFileName(sln_dir, p));
		annmdl_paths.FindAdd(p);
	}
	annmdl_paths.FindAdd(AnnMdl::PathFromAnnlay(annlay_path));

	int total_added = 0;
	int total_already = 0;
	bool any_error = false;

	for(int fi = 0; fi < annmdl_paths.GetCount(); fi++) {
		String annmdl_path = annmdl_paths[fi];
		if(!FileExists(annmdl_path)) {
			Cout() << label << ": skip missing: " << annmdl_path << "\n";
			continue;
		}

		// Load with full blob data so we can copy entries
		AnnMdl mdl;
		if(!mdl.LoadPath(annmdl_path, true)) {
			Cout() << label << ": failed to load: " << annmdl_path << "\n";
			any_error = true;
			continue;
		}

		Cout() << label << ": checking " << annmdl_path << " (" << mdl.entries.GetCount() << " entries)\n";

		int added = 0;
		int already = 0;

		for(int ri = 0; ri < role_to_canonical.GetCount(); ri++) {
			String role     = role_to_canonical.GetKey(ri);
			String canon_id = role_to_canonical[ri];

			// Check if canonical key already has a model entry with blobs
			const AnnMdlEntry* canon_e = mdl.FindEntry(canon_id);
			bool canon_has_blobs = canon_e &&
				(!canon_e->net_data.IsEmpty() || !canon_e->net_ref.IsEmpty() ||
				 !canon_e->session_data.IsEmpty() || !canon_e->session_ref.IsEmpty() ||
				 canon_e->session_offset >= 0);

			if(canon_has_blobs) {
				Cout() << "  " << canon_id << ": already present\n";
				already++;
				continue;
			}

			// Try to find a donor: first look through legacy aliases for this canonical group
			Vector<String> aliases = reg.LegacyAliases(canon_id);
			String donor_slot_id;
			for(int ai = 0; ai < aliases.GetCount() && donor_slot_id.IsEmpty(); ai++) {
				const AnnMdlEntry* ae = mdl.FindEntry(aliases[ai]);
				if(ae && (!ae->net_data.IsEmpty() || !ae->net_ref.IsEmpty() ||
				          !ae->session_data.IsEmpty() || !ae->session_ref.IsEmpty() ||
				          ae->session_offset >= 0))
					donor_slot_id = aliases[ai];
			}

			// If no alias donor, try any per-slot entry with matching suffix
			if(donor_slot_id.IsEmpty())
				donor_slot_id = FindPerSlotDonor(mdl, role);

			if(donor_slot_id.IsEmpty()) {
				Cout() << "  " << canon_id << ": no donor found (role=" << role << ") -- skipped\n";
				continue;
			}

			const AnnMdlEntry* donor = mdl.FindEntry(donor_slot_id);
			Cout() << "  " << canon_id << ": will copy from \"" << donor_slot_id << "\"\n";
			if(!dry_run) {
				AnnMdlEntry& new_e = mdl.GetOrAdd(canon_id);
				CopyEntryBlobs(new_e, *donor);
			}
			added++;
		}

		Cout() << "  result: " << added << " added, " << already << " already present\n";

		if(added > 0 && !dry_run) {
			String backup_path = annmdl_path + ".bak";
			if(!FileExists(backup_path)) {
				if(!FileCopy(annmdl_path, backup_path)) {
					Cout() << label << ": failed to create backup: " << backup_path << "\n";
					any_error = true;
					continue;
				}
			}
			if(!mdl.SavePath(annmdl_path)) {
				Cout() << label << ": failed to save: " << annmdl_path << "\n";
				any_error = true;
				continue;
			}
			Cout() << "  saved: " << annmdl_path << "\n";
		}

		total_added  += added;
		total_already += already;
	}

	if(dry_run)
		Cout() << label << ": done (dry-run) total_would_add=" << total_added << " total_already=" << total_already << "\n";
	else
		Cout() << label << ": done total_added=" << total_added << " total_already=" << total_already << "\n";

	return !any_error;
}

} // namespace Upp
