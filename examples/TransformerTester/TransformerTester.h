#ifndef _TransformerTester_TransformerTester_h
#define _TransformerTester_TransformerTester_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

#define LAYOUTFILE <TransformerTester/TransformerTester.lay>
#include <CtrlCore/lay.h>

#define IMAGECLASS TransformerImg
#define IMAGEFILE <TransformerTester/TransformerTester.iml>
#include <Draw/iml_header.h>

// Custom control to visualize attention weights
class TransformerAttentionControl : public ParentCtrl {
private:
    Vector<Vector<double>> attentionData;

public:
    TransformerAttentionControl();
    virtual void Paint(Draw& draw);
    
    void SetAttentionData(const Vector<Vector<double>>& data);
    Vector<Vector<double>> GetAttentionData() const { return attentionData; }
    
    typedef TransformerAttentionControl CLASSNAME;
};

// Custom control to display performance metrics
class TransformerMetricsControl : public ParentCtrl {
private:
    double loss;
    double accuracy;
    int trainingSteps;
    double tokensPerSecond;

public:
    TransformerMetricsControl();
    virtual void Paint(Draw& draw);
    
    void UpdateMetrics(double newLoss, double newAccuracy, 
                      int newTrainingSteps, double newTokensPerSec);
    void ClearMetrics();
    
    typedef TransformerMetricsControl CLASSNAME;
};

class TransformerApp : public WithTransformerTesterLayout<TopWindow> {
private:
    std::unique_ptr<TransformerCRTP> transformer;
    
    // Custom visualization controls
    TransformerAttentionControl attentionVis;
    TransformerMetricsControl modelMetrics;
    
    // Data controls
    EditField inputEdit;
    EditField outputEdit;
    
    // Action controls
    Button trainBtn;
    Button testBtn;
    Button visualizeBtn;

public:
    typedef TransformerApp CLASSNAME;
    TransformerApp();
    
    void InitModel();
    void OnTrain();
    void OnTest();
    void OnVisualize();
    
    // Accessor methods
    TransformerAttentionControl& GetAttentionVis() { return attentionVis; }
    TransformerMetricsControl& GetMetrics() { return modelMetrics; }
    EditField& GetInputEdit() { return inputEdit; }
    EditField& GetOutputEdit() { return outputEdit; }
};

#endif