#include "CLIPTester.h"

CLIPApp::CLIPApp() {
    Title("CLIP (Contrastive Language-Image Pre-training) Demo");
    Sizeable().MaximizeBox().MinimizeBox();

    // Set up the layout
    // manual layout

    // Initialize components
    // imageLabel.SetLabel("Image:");
    // textLabel.SetLabel("Text Prompt:");
    // similarityLabel.SetLabel("Similarity Scores:");
    // imageUrlLabel.SetLabel("Image URL:");
    // loadButton.SetLabel("Load Image");
    // computeButton.SetLabel("Compute Similarity");
    // embedImageBtn.SetLabel("Encode Image");
    // embedTextBtn.SetLabel("Encode Text");
    // compareBtn.SetLabel("Compare Embeddings");

    // Add controls to the layout
    Add(CtrlLayout(controls));

    // Initialize the CLIP model
    InitModel();

    // Set up the UI event handlers
    loadButton <<= THISBACK(OnLoadImage);
    computeButton <<= THISBACK(OnComputeSimilarity);
    embedImageBtn <<= THISBACK(OnEmbedImage);
    embedTextBtn <<= THISBACK(OnEmbedText);
    compareBtn <<= THISBACK(OnCompare);
}

void CLIPApp::InitModel() {
    // Initialize model components
    // In a real implementation, this would create the actual vision and text encoders
    // For now, we'll just set a flag to indicate the model is "loaded"
    model_loaded = true;
    
    LOG("CLIP model initialized with vision and text encoders");
    
    // Display initial placeholder content
    textPromptEdit.SetText("A photo of a cat");
}

void CLIPApp::OnLoadImage() {
    String file_path = SelectFileOpen("Image files\t*.jpg;*.jpeg;*.png;*.bmp\nAll files\t*.*");
    if (file_path.IsEmpty()) return;

    if (!FileExists(file_path)) {
        PromptOK("File does not exist");
        return;
    }

    try {
        Image img = StreamRaster::LoadFileAny(file_path);
        imageDisplay.SetImage(img);
        imageUrlEdit.SetText(file_path);
        Refresh();
    }
    catch (Exc e) {
        PromptOK("Error loading image: " + e);
    }
}

void CLIPApp::OnEmbedImage() {
    if (!model_loaded) {
        PromptOK("Model not loaded yet!");
        return;
    }

    String image_path = imageUrlEdit.GetText().ToString();
    if (image_path.IsEmpty()) {
        PromptOK("Please load an image first");
        return;
    }

    // In a real implementation, this would:
    // 1. Preprocess the image (resize, normalize, etc.)
    // 2. Pass it through the vision encoder
    // 3. Output the resulting embedding vector

    // For demonstration, we'll simulate the embedding process
    PromptOK("Image encoded to embedding vector (simulated)");

    LOG("CLIP: Image encoded to embedding vector");
}

void CLIPApp::OnEmbedText() {
    if (!model_loaded) {
        PromptOK("Model not loaded yet!");
        return;
    }

    String text_prompt = textPromptEdit.GetText().ToString();
    if (text_prompt.IsEmpty()) {
        PromptOK("Please enter a text prompt");
        return;
    }

    // In a real implementation, this would:
    // 1. Tokenize the text
    // 2. Pass it through the text encoder (Transformer)
    // 3. Output the resulting embedding vector

    // For demonstration, we'll simulate the embedding process
    PromptOK("Text encoded to embedding vector (simulated)");

    LOG("CLIP: Text encoded to embedding vector");
}

void CLIPApp::OnCompare() {
    if (!model_loaded) {
        PromptOK("Model not loaded yet!");
        return;
    }

    String image_path = imageUrlEdit.GetText().ToString();
    String text_prompt = textPromptEdit.GetText().ToString();

    if (image_path.IsEmpty()) {
        PromptOK("Please load an image first");
        return;
    }

    if (text_prompt.IsEmpty()) {
        PromptOK("Please enter a text prompt");
        return;
    }

    // In a real implementation, this would:
    // 1. Get image embedding (or compute if not cached)
    // 2. Get text embedding (or compute if not cached)
    // 3. Compute cosine similarity between embeddings
    // 4. Return similarity score

    // For demonstration, we'll simulate similarity scores
    VectorMap<String, double> similarities;
    similarities.Add("cat", 0.89);
    similarities.Add("dog", 0.23);
    similarities.Add("bird", 0.15);
    similarities.Add("car", 0.05);
    similarities.Add("airplane", 0.02);

    similarityDisplay.SetSimilarities(similarities);

    PromptOK(Format("Computed similarity for image and text: '%s'", text_prompt));
}

void CLIPApp::OnComputeSimilarity() {
    // This is a higher-level function that combines embedding and comparison
    OnEmbedImage();
    OnEmbedText();
    OnCompare();
}

// Custom control for image display
CLIPImageControl::CLIPImageControl() {
    SetLabel("Image Display");
    image = Null;
    imagePath = "";
}

void CLIPImageControl::Paint(Draw& draw) {
    Size sz = GetSize();
    draw.DrawRect(sz, SColorFace());

    if (image && !image.IsNull()) {
        // Draw the image scaled to fit the control
        Size imgSz = image.GetSize();
        double scale = min((double)sz.cx / imgSz.cx, (double)sz.cy / imgSz.cy);
        
        int newWidth = (int)(imgSz.cx * scale);
        int newHeight = (int)(imgSz.cy * scale);
        
        int offsetX = (sz.cx - newWidth) / 2;
        int offsetY = (sz.cy - newHeight) / 2;
        
        draw.DrawImage(offsetX, offsetY, newWidth, newHeight, image);
    } else {
        // Draw placeholder text
        draw.DrawText(10, 10, "No image loaded", StdFont(), Black());
    }
}

void CLIPImageControl::SetImage(const Image& img) {
    image = img;
    Refresh();
}

void CLIPImageControl::SetImagePath(const String& path) {
    imagePath = path;
    try {
        image = StreamRaster::LoadFileAny(path);
    }
    catch (Exc e) {
        LOG("Error loading image: " + e);
        image = Null;
    }
    Refresh();
}

// Custom control for text display
CLIPTextControl::CLIPTextControl() {
    SetLabel("Text Display");
    text = "Enter text prompt";
}

void CLIPTextControl::Paint(Draw& draw) {
    Size sz = GetSize();
    draw.DrawRect(sz, SColorFace());

    // Draw the text
    if (!text.IsEmpty()) {
        draw.DrawText(10, 10, text, StdFont(), Black());
    } else {
        draw.DrawText(10, 10, "No text", StdFont(), Black());
    }
}

void CLIPTextControl::SetText(const String& t) {
    text = t;
    Refresh();
}

// Custom control for similarity scores
CLIPSimilarityControl::CLIPSimilarityControl() {
    SetLabel("Similarity Scores");
}

void CLIPSimilarityControl::Paint(Draw& draw) {
    Size sz = GetSize();
    draw.DrawRect(sz, SColorFace());

    // Draw the similarity scores
    int y = 10;
    for (int i = 0; i < similarities.GetCount(); i++) {
        String label = similarities.GetKey(i);
        double score = similarities[i];
        
        String displayText = Format("%s: %.3f", label, score);
        draw.DrawText(10, y, displayText, StdFont(), Black());
        y += 20;
        
        // Draw a bar for the score
        int barWidth = (int)(score * (sz.cx - 40));  // Scale to 0-100% of available width
        draw.DrawRect(10, y-8, barWidth, 6, LtBlue());
        
        y += 15;  // Extra spacing
        
        if (y > sz.cy - 20) break;  // Don't overflow the control
    }
    
    if (similarities.GetCount() == 0) {
        draw.DrawText(10, 10, "No similarity data", StdFont(), Black());
    }
}

void CLIPSimilarityControl::SetSimilarities(const VectorMap<String, double>& sims) {
    similarities = sims;
    Refresh();
}