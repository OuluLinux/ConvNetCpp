#ifndef _GptTester_GptTester_h
#define _GptTester_GptTester_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <memory>

using namespace Upp;
using namespace ConvNet;

#define LAYOUTFILE <GptTester/GptTester.lay>
#include <CtrlCore/lay.h>

#define IMAGECLASS GptImg
#define IMAGEFILE <GptTester/GptTester.iml>
#include <Draw/iml_header.h>

// Custom control to visualize GPT attention patterns
class GptAttentionControl : public ParentCtrl {
private:
    Vector<Vector<double>> attentionData;
    String title;

public:
    GptAttentionControl();
    virtual void Paint(Draw& draw);
    
    void SetAttentionData(const Vector<Vector<double>>& data);
    void SetTitle(const String& t) { title = t; Refresh(); }
    Vector<Vector<double>> GetAttentionData() const { return attentionData; }
    
    typedef GptAttentionControl CLASSNAME;
};

// Custom control for displaying text generation
class GptTextDisplay : public Ctrl {
private:
    String generatedText;
    int cursorPosition;

public:
    GptTextDisplay();
    virtual void Paint(Draw& draw);
    
    void SetGeneratedText(const String& text);
    void AddText(const String& text);
    String GetText() const { return generatedText; }
    
    typedef GptTextDisplay CLASSNAME;
};

class GptApp : public TopWindow {
private:
    // GPT model and session (using regular Session instead of undefined GPTSession)
    std::unique_ptr<Session> gpt_session;
    
    // Controls
    EditField promptEdit;
    CtrlArea controls;
    Button generateBtn;
    Button clearBtn;
    GptTextDisplay textDisplay;
    
    // Hyperparameter controls
    SpinCtrl temperatureCtrl;
    SpinCtrl topKCtrl;
    CheckBox topKEnable;
    SpinCtrl nucleusPCtrl;
    CheckBox nucleusEnable;
    SpinCtrl maxTokensCtrl;
    
    // Model parameters
    int vocab_size = 1000;
    int embed_dim = 64;
    int num_heads = 4;
    int num_layers = 2;
    int ff_dim = 256;
    int max_seq_len = 50;

public:
    typedef GptApp CLASSNAME;
    GptApp();
    
    void InitModel();
    void OnGenerate();
    void OnClear();
    void OnHyperparametersChanged();
    
    // Accessor methods
    GptTextDisplay& GetTextDisplay() { return textDisplay; }
    EditField& GetPromptEdit() { return promptEdit; }
};

#endif