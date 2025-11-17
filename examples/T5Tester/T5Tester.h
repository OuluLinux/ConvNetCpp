#ifndef _T5Tester_T5Tester_h
#define _T5Tester_T5Tester_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

#define LAYOUTFILE <T5Tester/T5Tester.lay>
#include <CtrlCore/lay.h>

#define IMAGECLASS T5Img
#define IMAGEFILE <T5Tester/T5Tester.iml>
#include <Draw/iml_header.h>

// Custom control to visualize attention weights
class T5AttentionControl : public ParentCtrl {
private:
    Vector<Vector<double>> attentionData;

public:
    T5AttentionControl();
    virtual void Paint(Draw& draw);

    void SetAttentionData(const Vector<Vector<double>>& data);
    Vector<Vector<double>> GetAttentionData() const { return attentionData; }

    typedef T5AttentionControl CLASSNAME;
};

// Custom control to display performance metrics
class T5MetricsControl : public ParentCtrl {
private:
    double loss;
    double accuracy;
    int trainingSteps;
    double tokensPerSecond;

public:
    T5MetricsControl();
    virtual void Paint(Draw& draw);

    void UpdateMetrics(double newLoss, double newAccuracy,
                      int newTrainingSteps, double newTokensPerSec);
    void ClearMetrics();

    typedef T5MetricsControl CLASSNAME;
};

class T5App : public WithT5TesterLayout<TopWindow> {
private:
    // T5 model components - simplified for demonstration
    // In a real implementation, this would include encoder/decoder transformer modules
    std::unique_ptr<TransformerCRTP> encoder_model;
    std::unique_ptr<TransformerCRTP> decoder_model;

    // Custom visualization controls
    T5AttentionControl attentionVis;
    T5MetricsControl modelMetrics;

    // Data controls
    EditField inputEdit;
    EditField outputEdit;
    DropList taskSelect;  // For different T5 tasks like translation, summarization, etc.

    // Action controls
    Button trainBtn;
    Button testBtn;
    Button visualizeBtn;
    Button encodeBtn;
    Button decodeBtn;

public:
    typedef T5App CLASSNAME;
    T5App();

    void InitModel();
    void OnTrain();
    void OnTest();
    void OnVisualize();
    void OnEncode();
    void OnDecode();

    // Accessor methods
    T5AttentionControl& GetAttentionVis() { return attentionVis; }
    T5MetricsControl& GetMetrics() { return modelMetrics; }
    EditField& GetInputEdit() { return inputEdit; }
    EditField& GetOutputEdit() { return outputEdit; }
    DropList& GetTaskSelect() { return taskSelect; }
};

#endif