# MIDI Learning and Generation with Transformer Attention

## Overview
This pseudocode describes a MIDI learning application using modern transformer architecture with specialized attention mechanisms for music generation. The design incorporates advanced attention techniques suitable for musical content.

## Core Framework with Enhanced Attention

```pseudocode
// MIDI learning application using modern transformer architecture with attention
class MIDILearningWindow : extends BaseNeuralNetWindow {
    // MIDI specific components
    Array<String> midi_filenames;
    Button load_midi_btn, play_btn, stop_btn;
    SliderCtrl tempo_slider;
    Label current_song_label, status_label;
    
    // Piano roll visualizations
    PianoRollCtrl input_piano_roll;      // Shows original MIDI input
    PianoRollCtrl generated_piano_roll;  // Shows generated output
    Splitter piano_roll_splitter;        // Splits input and output piano rolls
    
    // Attention visualization
    AttentionVisualizerCtrl attention_visualizer;
    DropList attention_layer_dropdown;
    DropList attention_head_dropdown;
    
    // MIDI data management
    MIDIDataManager midi_manager;
    Vector<MIDISong> input_songs;
    MIDISong current_generated_song;
    
    MIDILearningWindow() {
        Title("MIDI Learning and Generation (Transformer Attention)");
        
        // Setup MIDI-specific UI components
        SetupMIDIComponents();
        
        // Override the base layout to include piano rolls and attention visualization
        SetupMIDILayout();
        
        // Load transformer network configuration with attention details
        network_config_editor.SetData(GetDefaultMIDITransformerConfig());
    }
    
    SetupMIDIComponents() {
        load_midi_btn.SetLabel("Load MIDI Files...");
        play_btn.SetLabel("Play");
        stop_btn.SetLabel("Stop");
        
        tempo_slider.MinMax(20, 240);  // BPM range
        tempo_slider.SetData(120);
        
        load_midi_btn <<= Callback(LoadMIDIFiles);
        play_btn <<= Callback(PlayCurrentSong);
        stop_btn <<= Callback(StopPlayback);
        tempo_slider <<= Callback(SetTempo);
        
        input_piano_roll.SetTitle("Input Songs");
        generated_piano_roll.SetTitle("Generated Song");
        
        // Setup attention visualization controls
        attention_layer_dropdown.SetLabel("Attention Layer:");
        attention_head_dropdown.SetLabel("Attention Head:");
        
        // Populate with values based on model configuration
        for (int i = 0; i < 8; i++) {  // Assuming 8 layers
            attention_layer_dropdown.Add("Layer " + i);
        }
        for (int i = 0; i < 8; i++) {  // Assuming 8 heads
            attention_head_dropdown.Add("Head " + i);
        }
        
        attention_layer_dropdown.SetIndex(0);
        attention_head_dropdown.SetIndex(0);
    }
    
    SetupMIDILayout() {
        // Create layout with controls at top, piano rolls in middle, and attention at right
        Splitter main_horizontal_splitter;
        main_horizontal_splitter.Horz();  // Horizontal split between content and attention
        
        // Left side: main content
        Splitter content_splitter;
        content_splitter.Vert();  // Vertical split for controls and piano rolls
        
        // Top: control panel from base class
        content_splitter << control_panel;
        
        // Middle: piano roll splitter
        piano_roll_splitter.Horz();  // Horizontal split for piano rolls
        piano_roll_splitter << input_piano_roll;
        piano_roll_splitter << generated_piano_roll;
        
        content_splitter << piano_roll_splitter;
        
        // Bottom: status and tempo controls
        ParentCtrl bottom_panel;
        bottom_panel.Add(current_song_label.LeftPos(0, 200).TopPos(0, 20));
        bottom_panel.Add(status_label.HSizePos(210, 150).TopPos(0, 20));
        bottom_panel.Add(tempo_slider.HSizePos(370, 0).TopPos(0, 20));
        
        content_splitter << bottom_panel;
        main_horizontal_splitter << content_splitter;
        
        // Right side: attention visualization
        Splitter attention_splitter;
        attention_splitter.Vert();
        attention_splitter.Add(attention_layer_dropdown.HSizePos().TopPos(0, 25));
        attention_splitter.Add(attention_head_dropdown.HSizePos().TopPos(30, 25));
        attention_splitter.Add(attention_visualizer.HSizePos().VSizePos(60, 0));
        
        main_horizontal_splitter << attention_splitter;
        
        Add(main_horizontal_splitter.SizePos());
    }
    
    GetDefaultMIDITransformerConfig() {
        // Modern transformer with attention mechanisms specifically for music
        return "{\n"
            "\t\"model_type\":\"transformer\",\n"
            "\t\"vocab_size\": 388,\n"              // MIDI vocab: 128 notes * 3 (on/off/duration) + special tokens\n"
            "\t\"n_positions\": 2048,\n"            // Maximum sequence length\n"
            "\t\"n_embd\": 512,\n"                 // Embedding dimension\n"
            "\t\"n_layer\": 8,\n"                  // Number of transformer layers\n" 
            "\t\"n_head\": 8,\n"                   // Number of attention heads\n"
            "\t\"attention_type\": \"causal\",\n"   // Causal attention for generation\n"
            "\t\"pos_encoding_type\": \"relative\",\n"  // Relative positional encoding\n"
            "\t\"use_velocity\": true,\n"           // Include velocity information\n"
            "\t\"use_timing\": true,\n"             // Include precise timing\n"
            "\t\"resid_pdrop\": 0.1,\n"            // Residual dropout\n"
            "\t\"attn_pdrop\": 0.1,\n"             // Attention dropout\n"
            "\t\"embd_pdrop\": 0.1,\n"             // Embedding dropout\n"
            "\t\"learning_rate\": 0.0003,\n"       // Learning rate\n"
            "\t\"optimizer\": \"adamw\",\n"         // Optimizer\n"
            "\t\"warmup_steps\": 5000,\n"          // Warmup steps for learning rate\n"
            "\t\"lr_decay\": true,\n"              // Use learning rate decay\n"
            "\t\"attention_dropout\": 0.1,\n"      // Attention-specific dropout\n"
            "\t\"residual_dropout\": 0.1,\n"       // Residual connection dropout\n"
            "\t\"mask_strategy\": \"causal_bar_level\"\n"  // Musical structure-aware masking\n"
            "}";
    }
    
    LoadMIDIFiles() {
        Array<String> files = SelectMultipleFilesOpen("MIDI files\t*.mid;*.midi\nAll files\t*.*");
        if (!files.IsEmpty()) {
            input_songs.Clear();
            midi_filenames.Clear();
            
            for each file in files {
                MIDISong song = midi_manager.LoadMIDIFile(file);
                input_songs.Add(song);
                midi_filenames.Add(file);
            }
            
            // Update the input piano roll visualization
            input_piano_roll.SetSongs(input_songs);
            
            // Reload the network with the new data
            ReloadNetwork();
        }
    }
    
    PlayCurrentSong() {
        // Play either input or generated song based on selection
        if (current_generated_song.IsValid()) {
            midi_manager.PlayMIDISong(current_generated_song, tempo_slider.GetData());
        } else if (!input_songs.IsEmpty()) {
            midi_manager.PlayMIDISong(input_songs[0], tempo_slider.GetData());  // Play first song
        }
    }
    
    StopPlayback() {
        midi_manager.StopPlayback();
    }
    
    SetTempo() {
        midi_manager.SetTempo(tempo_slider.GetData());
    }
    
    // Override training loop to handle MIDI-specific transformer training
    TrainingLoop() {
        while (running) {
            if (!paused && !input_songs.IsEmpty()) {
                // Train on MIDI sequences in batches
                for each song in input_songs {
                    TrainOnMIDISong(song);
                }
                
                // Generate a new song periodically
                if (session.GetIteration() % 200 == 0) {  // Every 200 iterations
                    GenerateNewSong();
                }
            }
            Sleep(10);  // Control training speed
        }
        stopped = true;
    }
    
    TrainOnMIDISong(MIDISong song) {
        // Convert MIDI song to transformer-compatible token sequence with attention considerations
        Vector<int> token_sequence = midi_manager.ConvertToTokenSequence(song);
        
        // Process in chunks if sequence is too long
        int chunk_size = 1024; // Max context length
        for (int i = 0; i < token_sequence.GetCount(); i += chunk_size) {
            int end_idx = min(i + chunk_size, token_sequence.GetCount());
            Vector<int> chunk = token_sequence.GetPart(i, end_idx - i);
            
            // Create input and target sequences (teacher forcing)
            Vector<int> input_tokens = chunk.GetPart(0, chunk.GetCount() - 1);
            Vector<int> target_tokens = chunk.GetPart(1, chunk.GetCount() - 1);
            
            // Convert to volumes and train
            Volume input_vol = ConvertTokensToVolume(input_tokens);
            Volume target_vol = ConvertTokensToVolume(target_tokens);
            
            session.TrainOnce(input_vol, target_vol);
        }
    }
    
    GenerateNewSong() {
        // Use the trained transformer to generate a new song using causal attention
        current_generated_song = midi_manager.GenerateSongWithTransformer(session, 512);  // Generate 512 tokens
        generated_piano_roll.SetSong(current_generated_song);
        
        status_label.SetLabel("Generated new song at iteration: " + session.GetIteration());
    }
    
    // Override refresh UI to update all visualizations
    RefreshUI() {
        // Update base visualizations
        layer_ctrl.Refresh();
        network_view.Refresh();
        
        // Update MIDI-specific visualizations
        input_piano_roll.Refresh();
        generated_piano_roll.Refresh();
        
        // Update attention visualization
        attention_visualizer.Refresh();
        
        // Update status
        status_label.SetLabel("Training Iteration: " + session.GetIteration() + 
                             ", Loss: " + session.GetLossAverage());
        
        PostCallback(RefreshUI);
    }
}

// Specialized visualization for MIDI piano rolls
class PianoRollCtrl : implements DataVisualizer {
    Vector<MIDISong> songs;
    MIDISong single_song;
    int visible_start_time, visible_end_time;
    int scroll_position;
    bool is_input_roll;  // True for input, False for generated
    String title;
    
    PianoRollCtrl(bool is_input) {
        is_input_roll = is_input;
        visible_start_time = 0;
        visible_end_time = 10000;  // 10 seconds default view
    }
    
    SetTitle(String t) {
        title = t;
    }
    
    SetSongs(Vector<MIDISong> s) {
        songs = s;
        Refresh();
    }
    
    SetSong(MIDISong s) {
        single_song = s;
        Refresh();
    }
    
    Paint(Draw& draw) {
        Size sz = GetSize();
        
        // Draw title
        draw.DrawText(10, 5, title, StdFont().Bold(), Black());
        
        // Draw background grid
        DrawGrid(draw, sz);
        
        // Draw MIDI notes as colored rectangles
        if (!songs.IsEmpty()) {
            for each song in songs {
                DrawMIDISong(draw, song, sz, is_input_roll ? Color(0, 100, 200) : Color(200, 50, 50));  // Blue for input, Red for generated
            }
        } else if (single_song.IsValid()) {
            DrawMIDISong(draw, single_song, sz, is_input_roll ? Color(0, 100, 200) : Color(200, 50, 50));
        }
        
        // Draw time indicator if playing
        DrawPlayhead(draw, sz);
    }
    
    DrawGrid(Draw& draw, Size sz) {
        // Draw horizontal lines for each MIDI note (128 total)
        for (int i = 0; i <= 128; i++) {
            int y = sz.cy - (i * sz.cy / 128);
            draw.DrawLine(0, y, sz.cx, y, i % 12 == 0 ? LightGray() : VeryLightGray());
        }
        
        // Draw time grid
        for (int i = 0; i <= 10; i++) {
            int x = i * sz.cx / 10;
            draw.DrawLine(x, 0, x, sz.cy, LightGray());
        }
    }
    
    DrawMIDISong(Draw& draw, MIDISong song, Size sz, Color note_color) {
        // Draw each note in the song as a rectangle
        for each note in song.GetNotes() {
            Rect note_rect = GetNoteRect(note, sz);
            // Use more sophisticated coloring based on velocity
            Color note_color_with_velocity = BlendColorWithVelocity(note_color, note.GetVelocity());
            draw.DrawRect(note_rect, note_color_with_velocity);
            
            // Draw note duration as the width of the rectangle
            draw.DrawRectContour(note_rect, Black());
        }
    }
    
    BlendColorWithVelocity(Color base_color, int velocity) {
        // Make notes with higher velocity brighter/more saturated
        int intensity = velocity / 127.0 * 100;  // 0-100%
        return Blend(base_color, White(), intensity/100.0);
    }
    
    GetNoteRect(MIDINote note, Size sz) {
        // Convert MIDI note to visual rectangle coordinates
        int x = MapTimeToX(note.GetStartTime(), sz);
        int width = MapTimeToX(note.GetDuration(), sz);
        int y = MapNoteToY(note.GetPitch(), sz);
        int height = sz.cy / 128;  // Fixed height per note
        
        return RectC(x, y, width, height);
    }
    
    MapTimeToX(int time, Size sz) {
        // Map time value to X coordinate in the control
        int duration = visible_end_time - visible_start_time;
        return (time - visible_start_time) * sz.cx / duration;
    }
    
    MapNoteToY(int pitch, Size sz) {
        // Map MIDI note (0-127) to Y coordinate (inverted: low notes at bottom)
        return sz.cy - (pitch * sz.cy / 128);
    }
    
    DrawPlayhead(Draw& draw, Size sz) {
        // Draw playhead line if playing
        // Implementation would show current position in the song
    }
    
    Refresh() {
        RefreshData();
    }
    
    SetSession(Session session) {
        // Not directly tied to session, but may display related info
    }
    
    MouseWheel(Point p, int zdelta, dword keyflags) {
        // Handle horizontal scrolling for time axis
        int time_range = visible_end_time - visible_start_time;
        int scroll_amount = time_range / 20;  // 5% of time range per scroll
        visible_start_time += (zdelta > 0) ? -scroll_amount : scroll_amount;
        visible_end_time += (zdelta > 0) ? -scroll_amount : scroll_amount;
        
        if (visible_start_time < 0) {
            visible_start_time = 0;
            visible_end_time = time_range;
        }
        
        Refresh();
    }
}

// Attention visualization for interpretability
class AttentionVisualizerCtrl : extends Ctrl {
    Array<Volume> attention_weights;  // Store attention weights for visualization
    int layer_idx, head_idx;
    Label layer_label, head_label;
    
    AttentionVisualizerCtrl() {
        layer_idx = 0;
        head_idx = 0;
        layer_label.SetLabel("Layer: 0");
        head_label.SetLabel("Head: 0");
    }
    
    SetAttentionWeights(Array<Volume> weights) {
        attention_weights = weights;
        Refresh();
    }
    
    SetLayer(int layer) {
        layer_idx = layer;
        layer_label.SetLabel("Layer: " + layer);
        Refresh();
    }
    
    SetHead(int head) {
        head_idx = head;
        head_label.SetLabel("Head: " + head);
        Refresh();
    }
    
    Paint(Draw& draw) {
        Size sz = GetSize();
        
        // Draw title
        draw.DrawText(5, 5, "Attention Heatmap", StdFont().Bold(), Black());
        layer_label.SetRect(5, 25, 100, 20);
        head_label.SetRect(5, 45, 100, 20);
        
        if (attention_weights.IsEmpty()) return;
        
        // Visualize attention weights as a heatmap
        // X-axis: query positions, Y-axis: key positions
        if (attention_weights.GetCount() > layer_idx && 
            attention_weights[layer_idx].GetDim(3) > head_idx) {
            Volume head_attention = GetAttentionHead(attention_weights[layer_idx], head_idx);
            
            int seq_len = head_attention.GetDim(0);  // Both query and key sequence length
            int cell_width = sz.cx / seq_len;
            int cell_height = sz.cy / seq_len;
            
            for (int pos_q = 0; pos_q < seq_len; pos_q++) {
                for (int pos_k = 0; pos_k < seq_len; pos_k++) {
                    float weight = head_attention.Get(pos_q, pos_k);
                    
                    // Color based on attention strength (darker = more attention)
                    int intensity = min(255, (int)(weight * 255));
                    Color c = RGB(intensity, intensity, intensity);
                    
                    int x = pos_k * cell_width;
                    int y = pos_q * cell_height;
                    
                    draw.DrawRect(RectC(x, y, cell_width, cell_height), c);
                }
            }
        }
    }
    
    GetAttentionHead(Volume layer_attention, int head_idx) {
        // Extract a specific attention head from the multi-head attention tensor
        // Implementation would extract the specified head
        return layer_attention;  // Simplified for pseudocode
    }
    
    Refresh() {
        RefreshData();
    }
}

// MIDI data management class with attention-aware tokenization
class MIDIDataManager {
    Vector<MIDISong> loaded_songs;
    MIDITokenizer tokenizer;
    
    MIDIDataManager() {
        tokenizer = MIDITokenizer();
    }
    
    MIDISong LoadMIDIFile(String filename) {
        // Load and parse MIDI file
        MIDISong song;
        // Implementation would use a MIDI parsing library
        return song;
    }
    
    void PlayMIDISong(MIDISong song, int tempo_bpm) {
        // Play the MIDI song using system MIDI player
    }
    
    void StopPlayback() {
        // Stop MIDI playback
    }
    
    void SetTempo(int bpm) {
        // Set playback tempo
    }
    
    Vector<int> ConvertToTokenSequence(MIDISong song) {
        // Convert MIDI song to token sequence for transformer with attention considerations
        return tokenizer.TokenizeSong(song);
    }
    
    MIDISong GenerateSongWithTransformer(Session& session, int length) {
        // Use trained transformer to generate a new MIDI song with causal attention
        MIDISong generated_song;
        
        // Start with start token
        Vector<int> input_tokens;
        input_tokens.Add(384);  // Start token
        
        // Generate token by token using the trained transformer with causal masking
        for (int i = 0; i < length; i++) {
            Volume input_vol = ConvertTokensToVolume(input_tokens);
            Vector<double> output = session.Predict(input_vol);
            
            // Sample next token from output distribution with temperature
            int next_token = SampleFromDistribution(output, 0.8);  // Temperature of 0.8
            
            // Convert token back to MIDI event
            if (next_token >= 0 && next_token <= 127) {
                // Note ON
                generated_song.AddNoteEvent(generated_song.GetDuration(), next_token, 100, 500); // pitch, velocity, duration
            } else if (next_token >= 128 && next_token <= 255) {
                // Note OFF
                generated_song.AddNoteOffEvent(generated_song.GetDuration(), next_token - 128);
            } else if (next_token >= 256 && next_token <= 383) {
                // Time shift
                generated_song.AddDeltaTime(next_token - 256);
            } else if (next_token == 385) {  // End token
                break;
            }
            
            // Add token to input for next prediction
            input_tokens.Add(next_token);
            if (input_tokens.GetCount() > 1024) {  // Max context length
                input_tokens.Remove(0);  // Remove oldest token (sliding window)
            }
        }
        
        return generated_song;
    }
    
    Volume ConvertTokensToVolume(Vector<int>& tokens) {
        // Convert token sequence to a volume tensor
        int seq_len = tokens.GetCount();
        Volume vol(seq_len, 1, 1);  // [seq_len, 1, 1] tensor for token IDs
        
        for (int i = 0; i < seq_len; i++) {
            vol.Set(i, 0, 0, tokens[i]);
        }
        
        return vol;
    }
    
    int SampleFromDistribution(Vector<double>& distribution, double temperature = 1.0) {
        // Sample from the probability distribution with temperature
        // Apply temperature to control randomness
        Vector<double> temp_dist = ApplyTemperature(distribution, temperature);
        
        double r = Randomf();
        double sum = 0.0;
        
        for (int i = 0; i < temp_dist.GetCount(); i++) {
            sum += temp_dist[i];
            if (r <= sum) {
                return i;
            }
        }
        
        return 0;  // Fallback
    }
    
    Vector<double> ApplyTemperature(Vector<double>& distribution, double temperature) {
        // Apply temperature scaling to the distribution
        Vector<double> scaled_dist = distribution;
        for (int i = 0; i < scaled_dist.GetCount(); i++) {
            scaled_dist[i] = pow(scaled_dist[i], 1.0 / temperature);
        }
        
        // Normalize
        double sum = Sum(scaled_dist);
        for (int i = 0; i < scaled_dist.GetCount(); i++) {
            scaled_dist[i] /= sum;
        }
        
        return scaled_dist;
    }
}

// Advanced MIDI tokenizer with attention considerations
class MIDITokenizer {
    // Standard MIDI tokens
    Map<String, int> note_on_tokens;     // 0-127 (note on events)
    Map<String, int> note_off_tokens;    // 128-255 (note off events) 
    Map<String, int> time_tokens;        // 256-383 (time shifts)
    Map<String, int> special_tokens;     // 384-387 (start, end, pad, mask)
    
    // Additional specialized tokens for attention
    Map<String, int> chord_tokens;       // Chord progressions
    Map<String, int> rhythm_tokens;      // Rhythmic patterns
    Map<String, int> instrument_tokens;  // Instrument changes
    
    // Constructor to initialize token mappings
    MIDITokenizer() {
        // Initialize standard MIDI tokens
        for (int i = 0; i < 128; i++) {
            note_on_tokens[Format("NOTE_ON_%d", i)] = i;
            note_off_tokens[Format("NOTE_OFF_%d", i)] = 128 + i;
        }
        
        for (int i = 0; i < 128; i++) {
            time_tokens[Format("TIME_%d", i)] = 256 + i;
        }
        
        special_tokens["START"] = 384;
        special_tokens["END"] = 385;
        special_tokens["PAD"] = 386;
        special_tokens["MASK"] = 387;
        
        // Initialize musical structure tokens
        chord_tokens["C_MAJOR"] = 388;
        chord_tokens["G_MAJOR"] = 389;
        chord_tokens["A_MINOR"] = 390;
        chord_tokens["E_MINOR"] = 391;
        // ... more chord tokens
        
        rhythm_tokens["QUARTER_NOTE"] = 400;
        rhythm_tokens["EIGHTH_NOTE"] = 401;
        rhythm_tokens["HALF_NOTE"] = 402;
        rhythm_tokens["WHOLE_NOTE"] = 403;
        // ... more rhythm tokens
    }
    
    // Tokenize with attention to musical structure
    Vector<int> TokenizeSong(MIDISong song) {
        Vector<int> tokens;
        
        // Add start token
        tokens.Add(special_tokens["START"]);
        
        // Group events by time to identify chords/rhythms
        Vector<Vector<MIDIEvent>> time_sliced_events = GroupByTime(song.GetEvents(), 10); // 10ms time slices
        
        for each time_slice in time_sliced_events {
            // Check for chords (multiple notes at same time)
            if (HasChord(time_slice)) {
                tokens.Add(chord_tokens[GetChordType(time_slice)]);
            }
            
            // Add individual notes
            for each event in time_slice {
                if (event.type == NOTE_ON) {
                    tokens.Add(note_on_tokens[GetNoteToken(event)]);
                } else if (event.type == NOTE_OFF) {
                    tokens.Add(note_off_tokens[GetNoteToken(event)]);
                }
            }
            
            // Add rhythm token if this time slice has notable rhythmic pattern
            if (HasRhythmPattern(time_slice)) {
                tokens.Add(rhythm_tokens[GetRhythmPattern(time_slice)]);
            }
        }
        
        // Add end token
        tokens.Add(special_tokens["END"]);
        
        return tokens;
    }
    
    Vector<Vector<MIDIEvent>> GroupByTime(Vector<MIDIEvent> events, int time_slice_ms) {
        // Group MIDI events by time slices
        Vector<Vector<MIDIEvent>> time_slices;
        int current_slice_start = 0;
        Vector<MIDIEvent> current_slice;
        
        for each event in events {
            if (event.time >= current_slice_start + time_slice_ms) {
                // Move to next slice
                if (!current_slice.IsEmpty()) {
                    time_slices.Add(current_slice);
                    current_slice.Clear();
                }
                current_slice_start = event.time;
            }
            current_slice.Add(event);
        }
        
        if (!current_slice.IsEmpty()) {
            time_slices.Add(current_slice);
        }
        
        return time_slices;
    }
    
    bool HasChord(Vector<MIDIEvent> events) {
        // Check if this time slice contains a chord (multiple simultaneous notes)
        int note_ons = 0;
        for each event in events {
            if (event.type == NOTE_ON) note_ons++;
        }
        return note_ons >= 3;  // Consider 3+ notes as a chord
    }
    
    String GetChordType(Vector<MIDIEvent> events) {
        // Determine chord type from note collection
        Vector<int> pitches;
        for each event in events {
            if (event.type == NOTE_ON) pitches.Add(event.pitch);
        }
        
        // Simplified chord detection
        Sort(pitches);
        
        // Convert to chord name (simplified)
        int root = pitches[0] % 12;
        if (root == 0) return "C_MAJOR";
        else if (root == 7) return "G_MAJOR";
        else if (root == 9) return "A_MINOR";
        else if (root == 4) return "E_MINOR";
        
        return "C_MAJOR";  // Default
    }
    
    bool HasRhythmPattern(Vector<MIDIEvent> events) {
        // Check for specific rhythmic patterns
        // Implementation would analyze timing patterns
        return events.GetCount() > 1;
    }
    
    String GetRhythmPattern(Vector<MIDIEvent> events) {
        // Return rhythm pattern name based on timing
        return "QUARTER_NOTE";  // Simplified
    }
    
    String GetNoteToken(MIDIEvent event) {
        return Format("NOTE_ON_%d", event.pitch);
    }
}

// Positional encoding for musical context
class MIDIPositionalEncoding {
    String encoding_type;
    bool use_learned, use_sinusoidal, use_relative;
    
    MIDIPositionalEncoding(String encoding_type) {
        this.encoding_type = encoding_type;
        
        if (encoding_type == "learned") {
            use_learned = true;
        } else if (encoding_type == "sinusoidal") {
            use_sinusoidal = true;
        } else if (encoding_type == "relative") {
            use_relative = true;
        }
    }
    
    Volume AddPositionalEncoding(Volume token_embeddings, int position) {
        if (use_learned) {
            // Learnable positional embeddings
            Volume pos_embedding = GetLearnedPositionEmbedding(position);
            return Add(token_embeddings, pos_embedding);
        } else if (use_sinusoidal) {
            // Fixed sinusoidal encoding
            return Add(token_embeddings, GetSinusoidalEncoding(position, token_embeddings.GetDim(-1)));
        } else if (use_relative) {
            // Relative positional encoding for better musical pattern recognition
            return Add(token_embeddings, GetRelativeEncoding(position, token_embeddings.GetDim(-1)));
        }
        
        return token_embeddings;
    }
    
    Volume GetRelativeEncoding(int position, int embedding_dim) {
        // Create relative positional encoding matrix
        // This helps capture musical patterns that are relative rather than absolute
        Vector<double> encoding(embedding_dim);
        
        for (int i = 0; i < embedding_dim; i += 2) {
            double angle = position / pow(10000, (2 * (i/2)) / embedding_dim);
            encoding[i] = sin(angle);
            if (i + 1 < embedding_dim) {
                encoding[i + 1] = cos(angle);
            }
        }
        
        return Volume::FromVector(encoding);
    }
    
    Volume GetLearnedPositionEmbedding(int position) {
        // Return learned positional embedding for this position
        // Implementation would have a learned embedding table
        return Volume();  // Simplified
    }
    
    Volume GetSinusoidalEncoding(int position, int embedding_dim) {
        // Standard sinusoidal positional encoding
        Vector<double> encoding(embedding_dim);
        
        for (int i = 0; i < embedding_dim; i++) {
            double angle = position / pow(10000, (2 * (i/2)) / embedding_dim);
            if (i % 2 == 0) {
                encoding[i] = sin(angle);
            } else {
                encoding[i] = cos(angle);
            }
        }
        
        return Volume::FromVector(encoding);
    }
}
```

## Attention Mechanisms for MIDI Applications

### 1. **Multi-Head Self-Attention** 
- Each head focuses on different musical aspects (rhythm, harmony, melody)
- Standard scaled dot-product attention with query, key, and value projections

### 2. **Causal/Self-Masked Attention** 
- Prevents attending to future tokens during generation
- Essential for autoregressive generation

### 3. **Positional Encodings**
- **Relative Positional Encodings**: Better for capturing musical patterns
- **Temporal Encodings**: Specifically for timing information in music

### 4. **Musical Structure-Aware Attention**
- **Bar-Level Attention**: Attend within musical bars for structural consistency
- **Chord-Based Attention**: Focus on harmonic relationships
- **Rhythm-Based Attention**: Focus on rhythmic patterns

### 5. **Multi-Modal Attention**
- Separate attention for pitch, timing, and velocity information
- Could have pitch attention, rhythm attention, and harmonic attention heads

## Key Advantages of This Approach

1. **Musical Structure Awareness**: Attention mechanisms specifically designed for musical patterns
2. **Interpretability**: Visual attention heatmap to understand what the model is focusing on
3. **Modern Architecture**: Transformer-based model instead of outdated RNN/LSTM
4. **Efficiency Considerations**: Options for local attention for longer sequences
5. **Comprehensive Visualization**: Piano roll views for both input and generated content

This design combines the common neural network visualization framework with modern transformer attention mechanisms specifically tailored for musical content generation and understanding.