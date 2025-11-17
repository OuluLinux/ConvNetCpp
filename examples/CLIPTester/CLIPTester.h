#ifndef _CLIPTester_CLIPTester_h
#define _CLIPTester_CLIPTester_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

#define LAYOUTFILE <CLIPTester/CLIPTester.lay>
#include <CtrlCore/lay.h>

#define IMAGECLASS CLIPImg
#define IMAGEFILE <CLIPTester/CLIPTester.iml>
#include <Draw/iml_header.h>

// Custom control for image display
class CLIPImageControl : public ParentCtrl {
private:
    Image image;
    String imagePath;

public:
    CLIPImageControl();
    virtual void Paint(Draw& draw);

    void SetImage(const Image& img);
    void SetImagePath(const String& path);
    Image GetImage() const { return image; }

    typedef CLIPImageControl CLASSNAME;
};

// Custom control for text display
class CLIPTextControl : public ParentCtrl {
private:
    String text;

public:
    CLIPTextControl();
    virtual void Paint(Draw& draw);

    void SetText(const String& t);
    String GetText() const { return text; }

    typedef CLIPTextControl CLASSNAME;
};

// Custom control to display similarity scores
class CLIPSimilarityControl : public ParentCtrl {
private:
    VectorMap<String, double> similarities;  // Text -> Similarity score

public:
    CLIPSimilarityControl();
    virtual void Paint(Draw& draw);

    void SetSimilarities(const VectorMap<String, double>& sims);
    VectorMap<String, double> GetSimilarities() const { return similarities; }

    typedef CLIPSimilarityControl CLASSNAME;
};

class CLIPApp : public WithCLIPTesterLayout<TopWindow> {
private:
    // CLIP model components (vision and text encoders)
    // In a real implementation, this would include actual CLIP model components
    std::unique_ptr<Network> vision_encoder;
    std::unique_ptr<Network> text_encoder;

    // For now we'll just simulate the functionality
    bool model_loaded;

    // Custom visualization controls
    CLIPImageControl imageDisplay;
    CLIPTextControl textDisplay;
    CLIPSimilarityControl similarityDisplay;

    // Data controls
    EditField imageUrlEdit;
    EditField textPromptEdit;
    Button loadImageBtn;
    Button computeSimilarityBtn;
    Button embedImageBtn;
    Button embedTextBtn;
    Button compareBtn;

public:
    typedef CLIPApp CLASSNAME;
    CLIPApp();

    void InitModel();
    void OnLoadImage();
    void OnComputeSimilarity();
    void OnEmbedImage();
    void OnEmbedText();
    void OnCompare();

    // Accessor methods
    CLIPImageControl& GetImageDisplay() { return imageDisplay; }
    CLIPTextControl& GetTextDisplay() { return textDisplay; }
    CLIPSimilarityControl& GetSimilarityDisplay() { return similarityDisplay; }
    EditField& GetImageUrlEdit() { return imageUrlEdit; }
    EditField& GetTextPromptEdit() { return textPromptEdit; }
};

#endif