#ifndef _ConvNet_SessionData_h_
#define _ConvNet_SessionData_h_

#include "Net.h"

namespace ConvNet {

class SessionData {
	
protected:
	friend class Session;
	friend class Brain;
	
	Vector<VolumeDataBase*> data, test_data, result_data;
	Vector<double> mins, maxs;
	Vector<int> labels, test_labels;
	Vector<String> classes;
	int data_w, data_h, data_d, data_len;
	bool is_data_result;
	
public:
	typedef SessionData CLASSNAME;
	SessionData();
	~SessionData();
	
	void BeginData(int cls_count, int count, int column_count, int test_count=0) {BeginDataClass<VolumeDataBase>(cls_count, count, 1, 1, column_count, test_count);}
	void BeginData(int cls_count, int count, int width, int height, int depth, int test_count=0) {BeginDataClass<VolumeDataBase>(cls_count, count, width, height, depth, test_count);}
	void BeginDataResult(int result_length, int count, int column_count, int test_count=0) {BeginDataResult<VolumeDataBase>(result_length, count, 1, 1, column_count, test_count);}
	void EndData();
	void ClearData();
	
	VolumeDataBase& Get(int i) {return *data[i];}
	VolumeDataBase& GetTest(int i) {return *test_data[i];}
	VolumeDataBase& GetResult(int i) {return *result_data[i];}
	String GetClass(int i) const {return classes[i];}
	double GetData(int i, int col) const;
	double GetTestData(int i, int col) const;
	double GetMax(int col) const {return maxs[col];}
	double GetMin(int col) const {return mins[col];}
	int GetLabel(int i) const {return labels[i];}
	int GetTestLabel(int i) const {return test_labels[i];}
	int GetDataCount() const {return data.GetCount();}
	int GetTestCount() const {return test_data.GetCount();}
	int GetDataLength() const {return data_w * data_h * data_d;}
	int GetDataWidth() const {return data_w;}
	int GetDataHeight() const {return data_h;}
	int GetDataDepth() const {return data_d;}
	int GetClassCount() const {return classes.GetCount();}
	void GetUniformClassData(int per_class, Vector<VolumeDataBase*>& volumes, Vector<int>& labels);
	
	SessionData& SetData(int i, int col, double value) {data[i]->Set(col, value); return *this;}
	SessionData& SetResult(int i, int col, double value) {result_data[i]->Set(col, value); return *this;}
	SessionData& SetLabel(int i, int label) {labels[i] = label; return *this;}
	SessionData& SetTestData(int i, int col, double value) {test_data[i]->Set(col, value); return *this;}
	SessionData& SetTestLabel(int i, int label) {test_labels[i] = label; return *this;}
	SessionData& SetClass(int i, const String& cls) {classes[i] = cls; return *this;}
	
	
	
	template <class T>
	void BeginDataClass(int cls_count, int count, int width, int height, int depth, int test_count=0) {
		ClearData();
		
		data_len = width * height * depth;
		data_w = width;
		data_h = height;
		data_d = depth;
		
		is_data_result = false;
		
		data.SetCount(count, NULL);
		for(int i = 0; i < data.GetCount(); i++)
			data[i] = new T(data_len);
		
		test_data.SetCount(test_count, NULL);
		for(int i = 0; i < test_data.GetCount(); i++)
			test_data[i] = new T(data_len);
		
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
	
	template <class T>
	void BeginDataResult(int result_length, int count, int width, int height, int depth, int test_count=0) {
		ClearData();
		
		data_len = width * height * depth;
		data_w = width;
		data_h = height;
		data_d = depth;
		
		is_data_result = true;
		
		data.SetCount(count, NULL);
		for(int i = 0; i < data.GetCount(); i++)
			data[i] = new T(data_len);
		
		result_data.SetCount(count, NULL);
		for(int i = 0; i < result_data.GetCount(); i++)
			result_data[i] = new T(result_length);
		
		test_data.SetCount(test_count, NULL);
		for(int i = 0; i < test_data.GetCount(); i++)
			test_data[i] = new T(data_len);
		
		mins.Clear();
		mins.SetCount(data_len, DBL_MAX);
		maxs.Clear();
		maxs.SetCount(data_len, -DBL_MAX);
		
	}
	
};

}

#endif
