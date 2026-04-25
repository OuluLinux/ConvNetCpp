#include "QualityTab.h"

NAMESPACE_UPP

QualityTab::QualityTab() {
	Add(list.SizePos());
	list.AddColumn("Metric");
	list.AddColumn("Value");
}

void QualityTab::UpdateMetrics(const Vector<Dataset>& datasets, const Vector<Category>& categories) {
	list.Clear();
	int total_suggestions = 0;
	int accepted_count = 0;
	int rejected_count = 0;
	VectorMap<int, int> cat_total;
	VectorMap<int, int> cat_accepted;

	for(const auto& ds : datasets) {
		for(const auto& ie : ds.images) {
			for(const auto& obj : ie.annotations) {
				if(!obj.accepted)
					continue;
				total_suggestions++;
				accepted_count++;
				cat_total.GetAdd(obj.category_id, 0)++;
				cat_accepted.GetAdd(obj.category_id, 0)++;
			}
			for(const auto& obj : ie.rejected_suggestions) {
				total_suggestions++;
				rejected_count++;
				cat_total.GetAdd(obj.category_id, 0)++;
			}
			total_suggestions += ie.suggestions.GetCount();
		}
	}

	list.Add("Total AI Suggestions", AsString(total_suggestions));
	list.Add("Accepted", AsString(accepted_count));
	list.Add("Rejected", AsString(rejected_count));
	double rate = (accepted_count + rejected_count) > 0
		? (double)accepted_count * 100.0 / (accepted_count + rejected_count)
		: 0;
	list.Add("Overall Acceptance Rate", Format("%.1f%%", rate));
	list.Add("", "");
	list.Add("Per-Category Acceptance", "");
	for(int i = 0; i < categories.GetCount(); i++) {
		int cid = categories[i].id;
		int total = cat_total.Get(cid, 0);
		int accepted = cat_accepted.Get(cid, 0);
		if(total <= 0)
			continue;
		double crate = (double)accepted * 100.0 / total;
		list.Add("  " + categories[i].name, Format("%.1f%% (%d/%d)", crate, accepted, total));
	}
}

END_UPP_NAMESPACE
