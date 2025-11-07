#ifndef _SimpleGAN_SimpleGAN_h
#define _SimpleGAN_SimpleGAN_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGEFILE <SimpleGAN/SimpleGAN.iml>
#include <Draw/iml_header.h>

double pdf1(double t);
double pdf2(double t);
double sample1();
double sample2();


#ifndef flagUSE_CONVNET
// a simple 2layer Net, except we have to be careful because our inputs are 1D scalars. Must be careful with init
class Net {
	
	struct Data {
		Vector<double> w1, b1, w2;
		double b2;
	};
	
public:
	void Init(int nh, double wscale) {
		this->nh = nh;
		this->wscale = wscale;
		
		
		// initialize parameters
		
		param.b2 = 0;
		grad_param.b2 = 0.5;
		
		for (int i = 0; i < nh; i++) {
			double c = (double)i / nh;
			double w = 14.7; // 10 * atanh(0.9)
			double b = -w * c;
			if (Randomf() < 0.5) {
				w = -w;
			}
			param.w1.Add(w);
			param.b1.Add(b);
			param.w2.Add((Randomf() - 0.5)*wscale);
			grad_param.w1.Add(0);
			grad_param.b1.Add(0);
			grad_param.w2.Add(0);
		}
		
		
		hcache.SetCount(nh, 0);

	}
	
	int nh = 0; // number of hidden units
	double t = 0.0;
	double wscale = 0.0;
	double lr = 0.00001;
	double reg = 0.00001;

	Data param, grad_param;
	Vector<double> hcache;
	
	double Forward(double t) {
		double x = param.b2;
		for (int i = 0; i < nh; i++) {
			double h = tanh(param.w1[i] * t + param.b1[i]);
			hcache[i] = h; // store for backward pass
			x += param.w2[i] * h;
		}
		this->t = t;
		return x;
	}
	
	double Backward(double dx) {
		grad_param.b2 += dx;
		double dt = 0;
		for (int i = 0; i < nh; i++) {
			double h = hcache[i];
			grad_param.w2[i] += h * dx;
			double dh = param.w2[i] * dx;
			double ds = (1.0 - h * h) * dh; // backprop through tanh
			grad_param.b1[i] += ds;
			grad_param.w1[i] += this->t * ds;
			dt += param.w1[i] * ds;
		}
		return dt;
	}
	
	void Update() {
		// dont learn 1st layer. 1D is finicky, lets just fix it
		//var p = this.param.w1; var g = this.grad_param.w1; for(var i=0;i<nh;i++) { p[i] += -lr * (g[i] + reg*p[i]); g[i] = 0; }
		//var p = this.param.b1; var g = this.grad_param.b1; for(var i=0;i<nh;i++) { p[i] += -lr * g[i]; g[i] = 0; }
		auto& p = param.w2;
		auto& g = grad_param.w2;
		for (int i = 0; i < nh; i++) {
			p[i] += -lr * (g[i] + reg * p[i]);
			g[i] = 0;
		}
		param.b2 += -lr * grad_param.b2;
		grad_param.b2 = 0;
	}
};
#endif


class SimpleGAN : public TopWindow {
	double (*pdf)(double);
	double (*sample)();
	
	#ifndef flagUSE_CONVNET
	::Net gen, disc;
	#else
	Session gen, disc;
	Volume tmp_input;
	Vector<double> tmp_ret;
	int input_width, input_height, input_depth;
	#endif
	
	// various vis hyperparams
	int orih = 350;
	int orix0 = 10;
	int orix1 = 390;
	int transh = 250;
	
	// various learning hyperparams
	double dt = 0.02;
	double lr = 0.0001;
	double reg = 0.00001;

	
	bool running = false, stopped = true;
	
public:
	typedef SimpleGAN CLASSNAME;
	SimpleGAN();
	~SimpleGAN() {running = false; while (!stopped) Sleep(100);}
	
	void Trainer();
	void Step();
	
	virtual void Paint(Draw& w);
	virtual void LeftDown(Point p, dword keyflags) {
		if (pdf == pdf1) {
			pdf = pdf2;
			sample = sample2;
		} else {
			pdf = pdf1;
			sample = sample1;
		}
	}

	void Refresh0() {Refresh();}
	

};

#endif
