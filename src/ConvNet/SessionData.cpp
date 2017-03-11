#include "ConvNet.h"

namespace ConvNet {

SessionData::SessionData() {
	data_w = 0;
	data_h = 0;
	data_d = 0;
	is_data_result = false;
	
}

SessionData::~SessionData() {
	ClearData();
}

void SessionData::ClearData() {
	for(int i = 0; i < data.GetCount(); i++) {
		delete data[i];
	}
	data.Clear();
	for(int i = 0; i < test_data.GetCount(); i++) {
		delete test_data[i];
	}
	test_data.Clear();
	for(int i = 0; i < result_data.GetCount(); i++) {
		delete result_data[i];
	}
	result_data.Clear();
	labels.Clear();
	test_labels.Clear();
	classes.Clear();
}

double SessionData::GetData(int i, int col) const {
	return data[i]->Get(col);
}

double SessionData::GetTestData(int i, int col) const {
	return test_data[i]->Get(col);
}

void SessionData::EndData() {
	for(int i = 0; i < data.GetCount(); i++) {
		VolumeDataBase* vol = data[i];
		if (!vol) {
			data.Remove(i);
			labels.Remove(i);
			i--;
			continue;
		}
		for(int j = 0; j < vol->GetCount(); j++) {
			double d = vol->Get(j);
			double& mind = mins[j];
			double& maxd = maxs[j];
			mind = min(mind, d);
			maxd = max(maxd, d);
		}
	}
	
	// Randomize data
	int count = data.GetCount() / 2;
	for(int i = 0; i < count; i++) {
		int a = Random(data.GetCount());
		int b = Random(data.GetCount());
		Swap(data[a],	data[b]);
		if (!is_data_result)
			Swap(labels[a],	labels[b]);
		else
			Swap(result_data[a], result_data[b]);
	}
	
}

void SessionData::GetUniformClassData(int per_class, Vector<VolumeDataBase*>& volumes, Vector<int>& labels) {
	ASSERT(per_class >= 0);
	Vector<int> counts;
	counts.SetCount(classes.GetCount(), 0);
	int remaining = per_class * classes.GetCount();
	volumes.SetCount(remaining);
	labels.SetCount(remaining);
	
	for(int i = 0; i < this->labels.GetCount() && remaining > 0; i++) {
		int label = this->labels[i];
		int& count = counts[label];
		if (count < per_class) {
			count++;
			remaining--;
			volumes[remaining] = data[i];
			labels[remaining] = label;
		}
	}
	
}

}
