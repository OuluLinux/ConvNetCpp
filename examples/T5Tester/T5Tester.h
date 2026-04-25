#include <Core/Core.h>
#ifndef _T5Tester_h
#define _T5Tester_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

class T5AttentionControl : public Label {
private:
    Vector<Vector<double>> attentionData;
public:
    void Init() {}
    T5AttentionControl();
    virtual void Paint(Draw& draw);
    void SetAttentionData(const Vector<Vector<double>>& data);
    Vector<Vector<double>> GetAttentionData() const { return clone(attentionData); }
    typedef T5AttentionControl CLASSNAME;
};

class T5MetricsControl : public ParentCtrl {
private:
    double loss, accuracy, tokensPerSecond;
    int trainingSteps;
public:
    void Init() {}
    T5MetricsControl();
    virtual void Paint(Draw& draw);
    void UpdateMetrics(double l, double a, int s, double t);
    void ClearMetrics();
    typedef T5MetricsControl CLASSNAME;
};

class T5App : public TopWindow {
public:
    std::unique_ptr<TransformerCRTP> encoder_model;
    std::unique_ptr<TransformerCRTP> decoder_model;
    T5AttentionControl attentionVis;
    T5MetricsControl modelMetrics;
    EditField seqInput, seqOutput, inputEdit, outputEdit;
    Label taskLabel, modelInfo;
    ParentCtrl controls;
    Button trainBtn, testBtn, visualizeBtn, encodeBtn, decodeBtn;
    DropList taskSelect;

public:
    void Init() {}
    typedef T5App CLASSNAME;
    T5App();
    void InitModel();
    void OnTrain(); void OnTest(); void OnVisualize(); void OnEncode(); void OnDecode();
};
#endif
