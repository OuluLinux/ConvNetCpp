#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

// sample random normal dist - from SimpleGAN example
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

// Custom SimpleGANNet class - from SimpleGAN example (simplified for testing)
class SimpleGANNet {

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
        grad_param.b2 = 0.5; // Initialize gradient

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

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("SimpleGAN custom Net test:");
    
    // Test the custom SimpleGANNet class implementation (as used in SimpleGAN)
    SimpleGANNet net;
    
    // Initialize with 10 hidden units and weight scale of 0.5
    net.Init(10, 0.5);
    
    // Test forward pass
    double input = 0.5;
    double output = net.Forward(input);
    
    LOG("  Forward pass - Input: " << input << ", Output: " << output);
    
    // Output should be a finite number
    ASSERT(std::isfinite(output));
    
    // Test backward pass
    double grad = 1.0;
    double input_grad = net.Backward(grad);
    
    LOG("  Backward pass - Grad: " << grad << ", Input grad: " << input_grad);
    
    // Input gradient should be finite
    ASSERT(std::isfinite(input_grad));
    
    // Test update
    net.Update();
    
    LOG("  Update step completed");
    
    // Run another forward pass to ensure the net still works after update
    double output2 = net.Forward(input);
    LOG("  Forward after update - Input: " << input << ", Output: " << output2);
    
    ASSERT(std::isfinite(output2));
    
    LOG("SimpleGAN tests completed successfully!");
}