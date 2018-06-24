#include "SimpleGAN.h"

// sample random normal dist
double RandomNormal() {
	double x1, x2, rad, y1;
	do {
		x1 = 2 * Randomf() - 1;
		x2 = 2 * Randomf() - 1;
		rad = x1 * x1 + x2 * x2;
	}
	while (rad >= 1 || rad == 0);
	double c = sqrt(-2 * log(rad) / rad);
	return x1 * c;
}

double Sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// 1bump gaussian
double mu = 0.5;
double sigma = 0.1;
double pdf1(double t) {
	double p = 1.0 / (sigma * sqrt(2 * M_PI)) * exp(-(t - mu) * (t - mu) / (2 * sigma * sigma)) * 50;
	return p;
}
double sample1() {
	return RandomNormal()*sigma + mu;
}

// 2bump gaussian
double mu1 = 0.25;
double sigma1 = 0.1;
double mu2 = 0.75;
double sigma2 = 0.1;
double pdf2(double t) {
	double p1 = 1.0 / (sigma * sqrt(2 * M_PI)) * exp(-(t - mu1) * (t - mu1) / (2 * sigma1 * sigma1)) * 50;
	double p2 = 1.0 / (sigma * sqrt(2 * M_PI)) * exp(-(t - mu2) * (t - mu2) / (2 * sigma2 * sigma2)) * 50;
	double p = 0.5 * p1 + 0.5 * p2;
	return p;
}

double sample2() {
	if (Randomf() < 0.5) {
		return RandomNormal()*sigma1 + mu1;
	}
	else {
		return RandomNormal()*sigma2 + mu2;
	}
}

#ifndef flagUSE_CONVNET



SimpleGAN::SimpleGAN()
{
	pdf = pdf1;
	sample = sample1;
	gen.Init(30, 0.5);
	disc.Init(30, 0.5);
	
	Title("Simple GAN");
	SetRect(0, 0, 400, 360);
	
	Thread::Start(THISBACK(Trainer));
}

void SimpleGAN::Trainer() {
	stopped = false;
	running = true;
	
	TimeStop ts;
	while (!Thread::IsShutdownThreads() && running) {
		
		Step();
		
		if (ts.Elapsed() >= 1000/60) {
			PostCallback(THISBACK(Refresh0));
			ts.Reset();
		}
	}
	
	stopped = true;
}

// do a single training step
void SimpleGAN::Step() {
	
	// forward and backward a generated (negative) example
	for (int k = 0; k < 5; k++) {
		double t = Randomf();
		double xgen = gen.Forward(t);
		double sgen = disc.Forward(xgen);
		double dsgen = Sigmoid(sgen);
		double dxgen = disc.Backward(dsgen);
		
		// forward and backward a real (positive) example
		double xdata = sample();
		double sdata = disc.Forward(xdata);
		double dsdata = Sigmoid(sdata) - 1;
		double dxdata = disc.Backward(dsdata);
		disc.Update();
	}
	// backward the generator
	double t = Randomf();
	double xgen = gen.Forward(t);
	double sgen = disc.Forward(xgen);
	double dsgen = Sigmoid(sgen);
	double dxgen = disc.Backward(dsgen);
	double dt = gen.Backward(-dxgen);
	gen.Update();
}

void SimpleGAN::Paint(Draw& w) {
	Size sz = GetSize();
	ImageDraw id(sz);
	
	id.DrawRect(sz, White);
	
	id.DrawLine(orix0, orih, orix1, orih, 2, Black);
	id.DrawLine(orix0, transh, orix1, transh, 2, Black);
	
	// draw the true distribution
	double t = 0;
	double prevp = 0;
	while (t <= 1) {
		double p = pdf(t);
		if (t > 0) {
			id.DrawLine(
				(t - dt)*(orix1 - orix0) + orix0,
				transh - prevp,
				t*(orix1 - orix0) + orix0,
				transh - p,
				2,
				Blue);
		}
		prevp = p;
		t += dt;
	}
	
	// draw the discriminator
	t = 0;
	prevp = 0;
	while (t <= 1) {
		double p = disc.Forward(t);
		p = Sigmoid(p) * 50; // Sigmoid
		if (t > 0) {
			id.DrawLine(
				(t - dt)*(orix1 - orix0) + orix0,
				transh - prevp,
				t*(orix1 - orix0) + orix0,
				transh - p,
				2,
				Green);
		}
		
		prevp = p;
		t += dt;
	}
	
	
	// draw the generating distribution arrows
	t = 0;
	while (t <= 1) {
		double x = gen.Forward(t);
		
		id.DrawLine(
			t*(orix1-orix0) + orix0,
			orih,
			x*(orix1-orix0) + orix0,
			transh,
			2,
			Red);
		
		t += dt;
	}
	
	w.DrawImage(0,0,id);
}

#endif
