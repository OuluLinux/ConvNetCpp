#ifndef _ConvNet_Mat_h_
#define _ConvNet_Mat_h_


namespace ConvNet {

class Mat : Moveable<Mat> {
	Vector<double> weight_gradients;
	Vector<double> weights;

protected:
	
	int width;
	int height;
	int length;
	
public:

	
	Mat();
	Mat(int width, int height, Mat& vol);
	Mat(int width, int height, const Vector<double>& weights);
	Mat(const Mat& o) {*this = o;}
	Mat(int width, int height); // Mat will be filled with random numbers
	Mat(int width, int height, double default_value);
	Mat(const Vector<double>& weights);
	Mat& Init(const Mat& v, double default_value=0.0) {return Init(v.GetWidth(), v.GetHeight(), default_value);}
	Mat& Init(int width, int height); // Mat will be filled with random numbers
	Mat& Init(int width, int height, const Vector<double>& weights);
	Mat& Init(int width, int height, double default_value);
	
	~Mat();
	
	void Serialize(Stream& s) {s % weight_gradients % weights % width % height % length;}
	
	Mat& operator=(const Mat& src);
	
	const Vector<double>& GetWeights() const {return weights;}
	const Vector<double>& GetGradients() const {return weight_gradients;}
	
	void Add(int i, double v);
	void Add(int x, int y, double v);
	void AddFrom(const Mat& volume);
	void AddFromScaled(const Mat& volume, double a);
	void AddGradient(int x, int y, double v);
	void AddGradient(int i, double v);
	void AddGradientFrom(const Mat& volume);
	double Get(int x, int y) const;
	double GetGradient(int x, int y) const;
	void Set(int x, int y, double v);
	void SetConst(double c);
	void SetConstGradient(double c);
	void SetGradient(int x, int y, double v);
	double Get(int i) const;
	void Set(int i, double v);
	double GetGradient(int i) const;
	void SetGradient(int i, double v);
	void ZeroGradients();
	void Store(ValueMap& map) const;
	void Load(const ValueMap& map);
	
	
	int GetPos(int x, int y) const;
	int GetWidth()  const {return width;}
	int GetHeight() const {return height;}
	int GetLength() const {return length;}
	int GetMaxColumn() const;
	int GetSampledColumn() const;
	
};

}

#endif
