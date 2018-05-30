#ifndef _ConvNet_SessionData_h_
#define _ConvNet_SessionData_h_

#include "Net.h"

namespace ConvNet {

class SessionData {
	
protected:
	friend class Session;
	friend class Brain;
	
	Vector<Vector<double> > data, test_data, result_data;
	Vector<double> mins, maxs;
	Vector<int> labels, test_labels;
	Vector<String> classes;
	int data_w, data_h, data_d, data_len;
	bool is_data_result;
	
public:
	typedef SessionData CLASSNAME;
	SessionData();
	~SessionData();
	
	void Serialize(Stream& s);
	
	void BeginData(int cls_count, int count, int column_count, int test_count=0) {BeginDataClass(cls_count, count, 1, 1, column_count, test_count);}
	void BeginData(int cls_count, int count, int width, int height, int depth, int test_count=0) {BeginDataClass(cls_count, count, width, height, depth, test_count);}
	void BeginDataResult(int result_length, int count, int column_count, int test_count=0) {BeginDataResult(result_length, count, 1, 1, column_count, test_count);}
	void EndData();
	void ClearData();
	
	Vector<double>& Get(int i) {return data[i];}
	Vector<double>& GetTest(int i) {return test_data[i];}
	Vector<double>& GetResult(int i) {return result_data[i];}
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
	void GetUniformClassData(int per_class, Vector<Vector<double> >& volumes, Vector<int>& labels);
	
	SessionData& SetData(int i, int col, double value) {data[i].Set(col, value); return *this;}
	SessionData& SetResult(int i, int col, double value) {result_data[i].Set(col, value); return *this;}
	SessionData& SetLabel(int i, int label) {labels[i] = label; return *this;}
	SessionData& SetTestData(int i, int col, double value) {test_data[i].Set(col, value); return *this;}
	SessionData& SetTestLabel(int i, int label) {test_labels[i] = label; return *this;}
	SessionData& SetClass(int i, const String& cls) {classes[i] = cls; return *this;}
	
	
	
	void BeginDataClass(int cls_count, int count, int width, int height, int depth, int test_count=0);
	void BeginDataResult(int result_length, int count, int width, int height, int depth, int test_count=0);
	
};

}

#endif
