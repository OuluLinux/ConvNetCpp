#include <Core/Core.h>
#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>

// Include llama.cpp headers
extern "C" {
    #include "src/llama.cpp/ggml/include/gguf.h"
    #include "src/llama.cpp/ggml/include/ggml.h"
    #include "src/llama.cpp/include/llama.h"
}

using namespace Upp;

#define MODELFILE <examples/GGUFGUI/GGUFGUI.lay>
#include <CtrlCore/Win32MainLoop.h>
#include "MODELFILE"

// Structure for AI parameters
struct AIParameters {
    double temperature = 0.8;
    int max_tokens = 512;
    int top_k = 40;
    double top_p = 0.9;
    double repeat_penalty = 1.1;
    int n_threads = 4;
    int n_batch = 8;
    int seed = -1;
};

// Simple chat message structure
struct ChatMessage {
    String role;  // "user" or "assistant"
    String content;
    Time timestamp;
};

// Main application class
class GGUFGUI : public WithGGUFGUI<Ctrl> {
public:
    typedef GGUFGUI CLASSNAME;
    GGUFGUI();
    ~GGUFGUI();

private:
    // Event handlers
    void ModelSelect();
    void LoadModel();
    void SendChatMessage();
    void UpdateParameterView();
    void AddChatMessage(String role, String content);
    
    // Parameter change handlers
    void TempSliderChanged();
    void TopPSliderChanged();
    void MaxTokensChanged();
    void TopKChanged();
    void RepeatPenaltyChanged();
    void NThreadsChanged();
    void NBatchChanged();
    void SeedChanged();
    
    // Llama integration methods
    bool InitializeLlama();
    void GenerateResponse(const String& user_input);
    void TokenCallback(llama_token token);
    
    // Data
    AIParameters params;
    Vector<ChatMessage> chat_history;
    String current_model_path;
    bool model_loaded = false;
    
    // Llama related objects
    struct llama_model *llama_model = nullptr;
    struct llama_context *llama_ctx = nullptr;
    struct llama_sampler *llama_sampler = nullptr;
    
    // UI elements
    ArrayCtrl chat_list;
};

void GGUFGUI::ModelSelect() {
    String model_path = SelectFile("Select GGUF model file (*.gguf)|*.gguf");
    if (!model_path.IsEmpty()) {
        model_file_edit <<= model_path;
        current_model_path = model_path;
    }
}

void GGUFGUI::LoadModel() {
    if (current_model_path.IsEmpty()) {
        PromptOK("Please select a model file first!");
        return;
    }
    
    // Check if the file exists
    if (!FileExists(current_model_path)) {
        PromptOK("Model file does not exist: " + current_model_path);
        return;
    }
    
    // Initialize llama with the selected model
    if (InitializeLlama()) {
        // Get model info for display
        int n_ctx = llama_n_ctx(llama_ctx);
        int n_vocab = llama_vocab_n_tokens(llama_model);
        int n_params = llama_model_n_params(llama_model);
        
        String info = "Model loaded successfully!\n";
        info << "Context size: " << n_ctx << "\n";
        info << "Vocabulary size: " << n_vocab << "\n";
        info << "Parameters: " << n_params;
        
        PromptOK(info);
        model_loaded = true;
        
        // Add system message to chat
        AddChatMessage("system", "Model '" + GetFileName(current_model_path) + "' loaded successfully.");
    } else {
        PromptOK("Failed to initialize GGUF model. Please check the console for details.");
        model_loaded = false;
    }
}

void GGUFGUI::SendChatMessage() {
    String user_input = message_input.GetData();
    if (user_input.IsEmpty()) {
        return;
    }
    
    // Add user message to chat
    AddChatMessage("user", user_input);
    
    // Clear input field
    message_input.Clear();
    
    if (!model_loaded || !llama_model || !llama_ctx) {
        AddChatMessage("system", "Please load a model first before sending messages.");
        return;
    }
    
    // Generate response using llama.cpp
    GenerateResponse(user_input);
}

bool GGUFGUI::InitializeLlama() {
    if (current_model_path.IsEmpty()) {
        return false;
    }
    
    // Free existing context if loaded
    if (llama_ctx) {
        llama_free(llama_ctx);
        llama_ctx = nullptr;
    }
    
    if (llama_model) {
        llama_model_free(llama_model);
        llama_model = nullptr;
    }
    
    // Load the model
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // Use CPU only for now
    model_params.vocab_only = false;
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    
    llama_model = llama_model_load_from_file(current_model_path, model_params);
    
    if (!llama_model) {
        LOG("ERROR: Failed to load model from: " + current_model_path);
        return false;
    }
    
    LOG("Model loaded successfully");
    
    // Initialize context
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048; // Context size
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads; // Use same threads for batch
    
    llama_ctx = llama_init_from_model(llama_model, ctx_params);
    
    if (!llama_ctx) {
        LOG("ERROR: Failed to initialize context");
        llama_model_free(llama_model);
        llama_model = nullptr;
        return false;
    }
    
    LOG("Context initialized successfully");
    
    return true;
}

void GGUFGUI::GenerateResponse(const String& user_input) {
    if (!llama_model || !llama_ctx) {
        AddChatMessage("system", "Model not initialized properly.");
        return;
    }
    
    // Create a simple prompt with the user input
    String prompt = "User: " + user_input + "\nAssistant:";
    std::string prompt_std = prompt.ToStd();

    // Tokenize the prompt
    std::vector<llama_token> tokens_list(prompt_std.length() + 1024); // Estimate required tokens
    int n_tokenized = llama_tokenize(llama_model, prompt_std.c_str(), tokens_list.data(), tokens_list.size(), true);
    
    if (n_tokenized < 0) {
        // Resize and tokenize again
        tokens_list.resize(-n_tokenized + 1024);
        n_tokenized = llama_tokenize(llama_model, prompt_std.c_str(), tokens_list.data(), tokens_list.size(), true);
    }

    if (n_tokenized <= 0) {
        AddChatMessage("system", "Failed to tokenize input.");
        return;
    }
    
    tokens_list.resize(n_tokenized);

    // Create batch for processing
    llama_batch batch = llama_batch_init(tokens_list.size(), 0, 1);
    for (int i = 0; i < n_tokenized; i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // Clear the KV cache for a new conversation
    llama_kv_cache_clear(llama_ctx);

    // Process the prompt through the model
    if (llama_decode(llama_ctx, batch) != 0) {
        AddChatMessage("system", "Failed to process input tokens.");
        llama_batch_free(batch);
        return;
    }

    // Prepare for text generation
    std::string response = "";
    const int max_tokens_response = params.max_tokens;
    int n_remain = max_tokens_response;
    
    // Create sampler based on parameters
    struct llama_sampler * sampler = llama_sampler_init(
        llama_sampler_chain_init(
            llama_sampler_temp_init(params.temperature),
            llama_sampler_top_k_init(params.top_k),
            llama_sampler_top_p_init(params.top_p),
            llama_sampler_typical_init(1.0f), // typical_p
            llama_sampler_tfs_init(1.0f),     // tfs_z
            llama_sampler_repeat_init(params.repeat_penalty, 64), // penalty_last_n
            nullptr // terminator
        )
    );

    // Generate tokens
    while (n_remain > 0) {
        // Get the last token's logits to sample from
        llama_token new_token_id = llama_sampler_sample(sampler, llama_ctx, -1);
        
        // Check for end of sequence token
        if (new_token_id == llama_token_eos(llama_model) || new_token_id == 0) {
            break;
        }

        // Convert token to string and append to response
        std::vector<char> buf(8); // Usually 8 chars is enough for most tokens
        int n_chars = llama_token_to_piece(llama_model, new_token_id, buf.data(), buf.size(), 0, true);
        if (n_chars > 0) {
            response += std::string(buf.data(), n_chars < 8 ? n_chars : 8);
        }

        // Add the new token to the batch for next iteration
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token_id, tokens_list.size() + (max_tokens_response - n_remain), { 0 }, true);
        
        // Decode the new token
        if (llama_decode(llama_ctx, batch) != 0) {
            break;
        }
        
        n_remain--;
    }

    // Clean up
    llama_sampler_free(sampler);
    llama_batch_free(batch);

    // Add the generated response to chat
    AddChatMessage("assistant", response.c_str());
}

void GGUFGUI::UpdateParameterView() {
    // Update the parameter display
    temp_slider.SetData(params.temperature);
    temp_value.SetLabel(DoubleStr(params.temperature, 2));
    
    max_tokens_ctrl.SetData(params.max_tokens);
    top_k_ctrl.SetData(params.top_k);
    top_p_slider.SetData(params.top_p);
    top_p_value.SetLabel(DoubleStr(params.top_p, 2));
    repeat_penalty_ctrl.SetData(params.repeat_penalty);
    n_threads_ctrl.SetData(params.n_threads);
    n_batch_ctrl.SetData(params.n_batch);
    seed_ctrl.SetData(params.seed);
}

void GGUFGUI::AddChatMessage(String role, String content) {
    ChatMessage msg;
    msg.role = role;
    msg.content = content;
    msg.timestamp = GetSysTime();
    
    chat_history.Add() = msg;
    
    // Update chat display with proper formatting
    String prefix = (role == "user") ? "[User] " : 
                   (role == "assistant") ? "[AI] " : 
                   (role == "system") ? "[System] " : 
                   "[" + role + "] ";
    
    String timestamp = Format("%02d:%02d:%02d", 
                              msg.timestamp.hour, 
                              msg.timestamp.minute, 
                              msg.timestamp.second);
    
    String formatted_msg = "[" + timestamp + "] " + prefix + content;
    
    // Add to chat display control
    String current_text = chat_display.GetData();
    if (!current_text.IsEmpty()) {
        current_text += "\n\n";
    }
    current_text += formatted_msg;
    chat_display.SetData(current_text);
    
    // Scroll to bottom
    chat_display.SetInsertPos(current_text.GetCount());
}

// Event handlers for parameter changes
void GGUFGUI::TempSliderChanged() {
    params.temperature = (double)temp_slider.GetData();
    temp_value.SetLabel(DoubleStr(params.temperature, 2));
}

void GGUFGUI::TopPSliderChanged() {
    params.top_p = (double)top_p_slider.GetData();
    top_p_value.SetLabel(DoubleStr(params.top_p, 2));
}

void GGUFGUI::MaxTokensChanged() {
    params.max_tokens = (int)max_tokens_ctrl.GetData();
}

void GGUFGUI::TopKChanged() {
    params.top_k = (int)top_k_ctrl.GetData();
}

void GGUFGUI::RepeatPenaltyChanged() {
    params.repeat_penalty = (double)repeat_penalty_ctrl.GetData();
}

void GGUFGUI::NThreadsChanged() {
    params.n_threads = (int)n_threads_ctrl.GetData();
}

void GGUFGUI::NBatchChanged() {
    params.n_batch = (int)n_batch_ctrl.GetData();
}

void GGUFGUI::SeedChanged() {
    params.seed = (int)seed_ctrl.GetData();
}

GGUFGUI::~GGUFGUI() {
    // Clean up llama resources
    if (llama_ctx) {
        llama_free(llama_ctx);
        llama_ctx = nullptr;
    }
    
    if (llama_model) {
        llama_model_free(llama_model);
        llama_model = nullptr;
    }
}

GGUFGUI::GGUFGUI() {
    // Initialize UI controls
    CtrlLayoutOKCancel(*this, "GGUF Model Chat Interface");
    
    // Connect events
    model_select_btn <<= THISBACK(ModelSelect);
    load_model_btn <<= THISBACK(LoadModel);
    send_btn <<= THISBACK(SendChatMessage);
    
    // Set up parameter controls and events
    temp_slider <<= THISBACK(TempSliderChanged);
    temp_slider.WhenAction = THISBACK(TempSliderChanged);
    max_tokens_ctrl <<= THISBACK(MaxTokensChanged);
    top_k_ctrl <<= THISBACK(TopKChanged);
    top_p_slider <<= THISBACK(TopPSliderChanged);
    top_p_slider.WhenAction = THISBACK(TopPSliderChanged);
    repeat_penalty_ctrl <<= THISBACK(RepeatPenaltyChanged);
    n_threads_ctrl <<= THISBACK(NThreadsChanged);
    n_batch_ctrl <<= THISBACK(NBatchChanged);
    seed_ctrl <<= THISBACK(SeedChanged);
    
    // Initialize parameter view
    UpdateParameterView();
    
    // Add initial system message
    AddChatMessage("system", "Welcome to the GGUF Model Chat Interface! Please load a model to begin chatting.");
}

GUI_APP_MAIN {
    GGUFGUI().Run();
}