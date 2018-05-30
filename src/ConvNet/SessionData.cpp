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

void SessionData::Serialize(Stream& s) {
	s % data % test_data % result_data
	  % mins % maxs
	  % labels % test_labels
	  % classes
	  % data_w % data_h % data_d % data_len
	  % is_data_result;
}

void SessionData::ClearData() {
	data.Clear();
	test_data.Clear();
	result_data.Clear();
	labels.Clear();
	test_labels.Clear();
	classes.Clear();
}

void SessionData::BeginDataClass(int cls_count, int count, int width, int height, int depth, int test_count) {
	ClearData();
	
	data_len = width * height * depth;
	data_w = width;
	data_h = height;
	data_d = depth;
	
	is_data_result = false;
	
	data.SetCount(count);
	for(int i = 0; i < data.GetCount(); i++)
		data[i].SetCount(data_len, 0);
	
	test_data.SetCount(test_count);
	for(int i = 0; i < test_data.GetCount(); i++)
		test_data[i].SetCount(data_len, 0);
	
	labels.SetCount(0);
	labels.SetCount(count, 0);
	test_labels.SetCount(0);
	test_labels.SetCount(test_count, -1);
	
	mins.Clear();
	mins.SetCount(data_len, DBL_MAX);
	maxs.Clear();
	maxs.SetCount(data_len, -DBL_MAX);
	
	classes.SetCount(cls_count);
	
}

void SessionData::BeginDataResult(int result_length, int count, int width, int height, int depth, int test_count) {
	ClearData();
	
	data_len = width * height * depth;
	data_w = width;
	data_h = height;
	data_d = depth;
	
	is_data_result = true;
	
	data.SetCount(count);
	for(int i = 0; i < data.GetCount(); i++)
		data[i].SetCount(data_len, 0);
	
	result_data.SetCount(count);
	for(int i = 0; i < result_data.GetCount(); i++)
		result_data[i].SetCount(result_length, 0);
	
	test_data.SetCount(test_count);
	for(int i = 0; i < test_data.GetCount(); i++)
		test_data[i].SetCount(data_len, 0);
	
	mins.Clear();
	mins.SetCount(data_len, DBL_MAX);
	maxs.Clear();
	maxs.SetCount(data_len, -DBL_MAX);
	
}

double SessionData::GetData(int i, int col) const {
	return data[i][col];
}

double SessionData::GetTestData(int i, int col) const {
	return test_data[i][col];
}

void SessionData::EndData() {
	for(int i = 0; i < data.GetCount(); i++) {
		Vector<double>& vol = data[i];
		if (!vol) {
			data.Remove(i);
			labels.Remove(i);
			i--;
			continue;
		}
		for(int j = 0; j < vol.GetCount(); j++) {
			double d = vol[j];
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

void SessionData::GetUniformClassData(int per_class, Vector<Vector<double> >& volumes, Vector<int>& labels) {
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
			HeaplessCopy(volumes[remaining], data[i]);
			labels[remaining] = label;
		}
	}
	
}

}
