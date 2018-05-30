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
	var c = sqrt(-2 * log(rad) / rad);
	return x1 * c;
}

double Sigmoid(x) {
	return 1.0 / (1.0 + Math.exp(-x));
}

// 1bump gaussian
double mu = 0.5;
double sigma = 0.1;
double pdf1(t) {
	double p = 1.0 / (sigma * Math.sqrt(2 * Math.PI)) * Math.exp(-(t - mu) * (t - mu) / (2 * sigma * sigma)) * 50;
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
double pdf2(t) {
	var p1 = 1.0 / (sigma * Math.sqrt(2 * Math.PI)) * Math.exp(-(t - mu1) * (t - mu1) / (2 * sigma1 * sigma1)) * 50;
	var p2 = 1.0 / (sigma * Math.sqrt(2 * Math.PI)) * Math.exp(-(t - mu2) * (t - mu2) / (2 * sigma2 * sigma2)) * 50;
	var p = 0.5 * p1 + 0.5 * p2;
	return p;
}
function sample2() {
	if (Math.random() < 0.5) {
		return RandomNormal()*sigma1 + mu1;
	}
	else {
		return RandomNormal()*sigma2 + mu2;
	}
}
var pdf = pdf1;
var sample = sample1;
var curd = 1;

// various vis hyperparams
var orih = 350;
var orix0 = 10;
var orix1 = 390;
var transh = 250;
// various learning hyperparams
var dt = 0.02;
var lr = 0.0001;
var reg = 0.00001;

// a simple 2layer Net, except we have to be careful because our inputs are 1D scalars. Must be careful with init
var Net = function(nh, wscale) {
	this.nh = nh; // number of hidden units
	// initialize parameters
	var param = {}
				var grad_param = {}
								 param.w1 = [];
	param.b1 = [];
	param.w2 = [];
	param.b2 = 0;
	grad_param.w1 = [];
	grad_param.b1 = [];
	grad_param.w2 = [];
	grad_param.b2 = 0.5;
	for (var i = 0;i < nh;i++) {
		var c = i / nh;
		var w = 14.7; // 10 * atanh(0.9)
		var b = -w * c;
		if (Math.random() < 0.5) {
			w = -w;
		}
		param.w1.push(w);
		param.b1.push(b);
		param.w2.push((Math.random() - 0.5)*wscale);
		grad_param.w1.push(0);
		grad_param.b1.push(0);
		grad_param.w2.push(0);
	}
	this.param = param;
	this.grad_param = grad_param;
	this.hcache = [];
	for (var i = 0;i < nh;i++) {
		this.hcache[i] = 0;
	}
}
Net.prototype = {
forward:
	function(t) {
		var param = this.param;
		var x = param.b2;
		for (var i = 0;i < this.nh;i++) {
			var h = Math.tanh(param.w1[i] * t + param.b1[i]);
			this.hcache[i] = h; // store for backward pass
			x += param.w2[i] * h;
		}
		this.t = t;
		return x;
	},
backward:
	function(dx) {
		var grad_param = this.grad_param;
		var param = this.param;
		grad_param.b2 += dx;
		var dt = 0;
		for (var i = 0;i < this.nh;i++) {
			var h = this.hcache[i]
					grad_param.w2[i] += h * dx;
			var dh = param.w2[i] * dx;
			var ds = (1.0 - h * h) * dh; // backprop through tanh
			grad_param.b1[i] += ds;
			grad_param.w1[i] += this.t * ds;
			dt += param.w1[i] * ds;
		}
		return dt;
	},
update:
	function() {
		// do param update and reset gradients to zero
		var nh = this.nh;
		// dont learn 1st layer. 1D is finicky, lets just fix it
		//var p = this.param.w1; var g = this.grad_param.w1; for(var i=0;i<nh;i++) { p[i] += -lr * (g[i] + reg*p[i]); g[i] = 0; }
		//var p = this.param.b1; var g = this.grad_param.b1; for(var i=0;i<nh;i++) { p[i] += -lr * g[i]; g[i] = 0; }
		var p = this.param.w2;
		var g = this.grad_param.w2;
		for (var i = 0;i < nh;i++) {
			p[i] += -lr * (g[i] + reg * p[i]);
			g[i] = 0;
		}
		this.param.b2 += -lr * this.grad_param.b2;
		this.grad_param.b2 = 0;
	}
}
var gen = new Net(30, 0.5); // generator
var disc = new Net(30, 0.5); // discriminator

// do a single training step
function step() {
	// forward and backward a generated (negative) example
	for (var k = 0;k < 5;k++) {
		var t = Math.random();
		var xgen = gen.forward(t);
		var sgen = disc.forward(xgen);
		var dsgen = Sigmoid(sgen);
		var dxgen = disc.backward(dsgen);
		// forward and backward a real (positive) example
		var xdata = sample();
		var sdata = disc.forward(xdata);
		var dsdata = Sigmoid(sdata) - 1;
		var dxdata = disc.backward(dsdata);
		disc.update();
	}
	// backward the generator
	var t = Math.random();
	var xgen = gen.forward(t);
	var sgen = disc.forward(xgen);
	var dsgen = Sigmoid(sgen);
	var dxgen = disc.backward(dsgen);
	var dt = gen.backward(-dxgen);
	gen.update();
}

function render() {

	var svg = d3.select('#svgvis');
	svg.html(''); // wipe
	
	svg.append('line').attr('x1', orix0).attr('y1', orih).attr('x2', orix1).attr('y2', orih).attr('stroke', '#000').attr('stroke-width', '2px');
	svg.append('line').attr('x1', orix0).attr('y1', transh).attr('x2', orix1).attr('y2', transh).attr('stroke', '#000').attr('stroke-width', '2px');
	
	// draw the true distribution
	var t = 0;
	var prevp = 0;
	while (t <= 1) {
		var p = pdf(t);
		if (t > 0) {
			svg.append('line').attr('x1', (t - dt)*(orix1 - orix0) + orix0).attr('y1', transh - prevp)
			.attr('x2', t*(orix1 - orix0) + orix0).attr('y2', transh - p)
			.attr('stroke', '#00F')
			.attr('stroke-width', '1px');
		}
		prevp = p;
		t += dt;
	}
	
	// draw the discriminator
	var t = 0;
	var prevp = 0;
	while (t <= 1) {
		var p = disc.forward(t);
		p = Sigmoid(p) * 50; // Sigmoid
		if (t > 0) {
			svg.append('line').attr('x1', (t - dt)*(orix1 - orix0) + orix0).attr('y1', transh - prevp)
			.attr('x2', t*(orix1 - orix0) + orix0).attr('y2', transh - p)
			.attr('stroke', '#0F0')
			.attr('stroke-width', '1px');
		}
		
