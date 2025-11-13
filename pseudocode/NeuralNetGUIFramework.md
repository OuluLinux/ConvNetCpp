# Neural Network GUI Framework for U++ ConvNetCpp

## Overview
This document describes a common GUI framework extracted from the U++ examples in ConvNetCpp.
It provides a standardized structure for creating new neural network visualization examples.

## Common Components

### 1. Base Window Classes
- `BaseNeuralNetWindow`: For simple layouts using TopWindow
- `BaseDockingNeuralNetWindow`: For complex layouts using DockWindow

### 2. Core Session Management
- `Session` object for neural network operations
- Training control: Start, Stop, Pause, Resume, Reset
- JSON network configuration loading

### 3. Visualization Components
- `LayerCtrl`: Shows neural network layer structure
- `PointCtrl`: Visualizes 2D classification data
- `HeatmapTimeView`: Shows network activations over time
- `TrainingGraph`: Displays training metrics
- `ConvLayerCtrl`: Displays convolutional layers

### 4. Layout Systems
- Splitter-based layouts (horizontal/vertical)
- Dockable window layouts with hide/resize capabilities

## Core Framework

```pseudocode
// Base class for neural network visualization windows
interface NeuralNetVisualizationWindow {
    // Core session management
    Session session;
    bool running, stopped, paused;
    
    // UI Components
    // Layout containers
    Splitter main_splitter;
    ParentCtrl control_panel;
    ParentCtrl network_view_panel;
    ParentCtrl data_view_panel;
    
    // Control elements
    Button start_btn, stop_btn, reset_btn, pause_btn, resume_btn;
    DocEdit network_config_editor;
    TrainingGraph training_graph;
    
    // Main UI elements
    LayerCtrl layer_ctrl;
    HeatmapTimeView network_view;
    
    // Core methods
    Initialize();                    // Initialize UI components and layouts
    StartTraining();                 // Start the training thread
    StopTraining();                  // Stop the training thread
    PauseTraining();                 // Pause training
    ResumeTraining();                // Resume training
    ReloadNetwork();                 // Reload network from JSON config
    UpdateVisualizations();          // Update all visualization components
    RefreshUI();                     // Refresh UI at regular intervals
}

// Abstract base class implementing the common pattern
abstract class BaseNeuralNetWindow : implements NeuralNetVisualizationWindow {
    Session session;
    bool running = false, stopped = true, paused = false;
    
    // UI Components
    Splitter main_splitter;
    ParentCtrl control_panel, network_view_panel, data_view_panel;
    Button start_btn, stop_btn, reset_btn, pause_btn, resume_btn;
    DocEdit network_config_editor;
    TrainingGraph training_graph;
    LayerCtrl layer_ctrl;
    HeatmapTimeView network_view;
    
    // Constructor
    BaseNeuralNetWindow() {
        SetupWindow();
        SetupUIComponents();
        SetupLayout();
        SetupCallbacks();
        
        PostCallback(RefreshUI);  // Start UI refresh loop
    }
    
    SetupWindow() {
        Title("Neural Network Visualization");
        Sizeable().MaximizeBox().MinimizeBox();
    }
    
    SetupUIComponents() {
        // Initialize all UI components
        InitializeButtons();
        InitializeNetworkConfigEditor();
        InitializeVisualizationCtrls();
    }
    
    abstract InitializeVisualizationCtrls();  // Different visualizations for different examples
    
    InitializeButtons() {
        start_btn.SetLabel("Start");
        stop_btn.SetLabel("Stop");
        reset_btn.SetLabel("Reset");
        pause_btn.SetLabel("Pause");
        resume_btn.SetLabel("Resume");
        
        // Connect callbacks
        start_btn <<= Callback(StartTraining);
        stop_btn <<= Callback(StopTraining);
        reset_btn <<= Callback(ResetTraining);
        pause_btn <<= Callback(PauseTraining);
        resume_btn <<= Callback(ResumeTraining);
    }
    
    InitializeNetworkConfigEditor() {
        network_config_editor.SetData(GetDefaultNetworkConfig());
    }
    
    abstract GetDefaultNetworkConfig();  // Different default configs for different examples
    
    SetupLayout() {
        // Main layout arrangement
        control_panel.Add(start_btn.LeftPos(0, 80).TopPos(0, 30));
        control_panel.Add(stop_btn.LeftPos(85, 80).TopPos(0, 30));
        control_panel.Add(reset_btn.LeftPos(170, 80).TopPos(0, 30));
        control_panel.Add(pause_btn.LeftPos(0, 80).TopPos(35, 30));
        control_panel.Add(resume_btn.LeftPos(85, 80).TopPos(35, 30));
        control_panel.Add(network_config_editor.HSizePos(0, 0).VSizePos(70, 30));
        
        // Add panels to main splitter
        main_splitter.Vert();  // Vertical split
        main_splitter << control_panel << network_view_panel;
        
        // Add main splitter to window
        Add(main_splitter.SizePos());
    }
    
    SetupCallbacks() {
        // All UI callbacks setup here
    }
    
    StartTraining() {
        if (running) StopTraining();
        running = true;
        stopped = false;
        ReloadNetwork();
        Thread::Start(TrainingLoop);
    }
    
    StopTraining() {
        running = false;
        while (!stopped) Sleep(100);  // Wait for thread to finish
    }
    
    PauseTraining() {
        paused = true;
    }
    
    ResumeTraining() {
        paused = false;
    }
    
    ResetTraining() {
        StopTraining();
        ReloadNetwork();
        StartTraining();
    }
    
    ReloadNetwork() {
        session.StopTraining();
        String config = network_config_editor.GetData();
        bool success = session.MakeLayers(config);
        if (success) {
            session.StartTraining();
        }
    }
    
    TrainingLoop() {
        while (running) {
            if (!paused) {
                session.TrainIteration();
            }
            // Sleep to control training speed
            Sleep(10);
        }
        stopped = true;
    }
    
    RefreshUI() {
        // Update visualization components
        layer_ctrl.Refresh();
        network_view.Refresh();
        
        // Post next refresh
        PostCallback(RefreshUI);
    }
}

// Docking window version for complex layouts
abstract class BaseDockingNeuralNetWindow : extends DockWindow implements NeuralNetVisualizationWindow {
    Session session;
    bool running = false, stopped = true, paused = false;
    
    // Dockable panels
    Dockable network_edit_dock;
    Dockable network_view_dock;
    Dockable training_graph_dock;
    
    // UI components
    ParentCtrl control_panel;
    DocEdit network_config_editor;
    Button start_btn, stop_btn, reload_btn;
    TrainingGraph training_graph;
    LayerCtrl layer_ctrl;
    HeatmapTimeView network_view;
    
    BaseDockingNeuralNetWindow() {
        Title("Neural Network Visualization");
        Sizeable().MaximizeBox().MinimizeBox().Zoomable();
        
        SetupUIComponents();
        SetupDockingLayout();
        SetupCallbacks();
        
        PostCallback(RefreshUI);
    }
    
    SetupUIComponents() {
        // Similar to base class but with dockable containers
    }
    
    SetupDockingLayout() {
        DockInit();
        
        // Create dockables
        control_panel.Add(network_config_editor.HSizePos().VSizePos(0, 30));
        control_panel.Add(reload_btn.HSizePos().BottomPos(0, 30));
        
        network_edit_dock = Dockable(control_panel, "Network Editor");
        network_view_dock = Dockable(network_view, "Network View");
        training_graph_dock = Dockable(training_graph, "Training Graph");
        
        // Add dockables to different positions
        AutoHide(DOCK_LEFT, network_edit_dock.SizeHint(Size(400, 300)));
        DockRight(network_view_dock.SizeHint(Size(300, 400)));
        DockBottom(training_graph_dock.SizeHint(Size(500, 200)));
    }
    
    virtual DockInit() {
        // Initialize docking behavior
    }
    
    // Other methods similar to base class...
}

// Framework for multi-AI visualization
class MultiAIOverviewCtrl : extends ParentCtrl {
    Array<NeuralNetVisualizationWindow> ai_instances;
    Splitter instances_splitter;
    Array<MiniVisualizationCtrl> mini_views;
    
    MultiAIOverviewCtrl(int num_instances) {
        for (i = 0; i < num_instances; i++) {
            ai_instances.Add(CreateAIInstance(i));
            mini_views.Add(CreateMiniViewFor(ai_instances[i]));
        }
        SetupOverviewLayout();
    }
    
    SetupOverviewLayout() {
        instances_splitter.Horz();
        for (each mini_view in mini_views) {
            instances_splitter << mini_view;
        }
        Add(instances_splitter.SizePos());
    }
    
    CreateAIInstance(index) {
        // Create and return specific AI instance
    }
    
    CreateMiniViewFor(instance) {
        // Create miniaturized view of the instance
    }
}

// Data visualization controls
interface DataVisualizer {
    SetSession(Session session);
    Refresh();
}

class PointCtrl : implements DataVisualizer {
    Session session;
    
    SetSession(Session ses) {
        session = ses;
    }
    
    Refresh() {
        // Draw 2D point data for classification
        RefreshData();
    }
    
    Paint(Draw& draw) {
        // Draw points and classification boundaries
    }
}

class TrainingGraph : implements DataVisualizer {
    Session session;
    Array<double> values;
    double min_val, max_val;
    
    AddValue(double value) {
        values.Add(value);
        if (values.GetCount() > MAX_POINTS) {
            values.Remove(0);  // Maintain window
        }
    }
    
    Paint(Draw& draw) {
        // Draw the training graph
    }
}
```

## Example Implementation

The framework can be specialized for specific neural network tasks:

### 2D Classification Example
```pseudocode
class Classification2DWindow : extends BaseNeuralNetWindow {
    PointCtrl data_view;
    Button circle_data_btn, spiral_data_btn;
    
    InitializeVisualizationCtrls() {
        layer_ctrl.SetSession(session);
        network_view.SetSession(session);
        data_view.SetSession(session);  // For 2D classification visualization
        
        // Add to panels
        network_view_panel.Add(layer_ctrl.HSizePos().VSizePos());
        data_view_panel.Add(data_view.HSizePos().VSizePos());
    }
    
    GetDefaultNetworkConfig() {
        return "[\n"
            "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":2},\n"
            "\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"tanh\"},\n"
            "\t{\"type\":\"fc\", \"neuron_count\":2, \"activation\": \"tanh\"},\n"
            "\t{\"type\":\"softmax\", \"class_count\":2},\n"
            "\t{\"type\":\"sgd\", \"learning_rate\":0.01, \"momentum\":0.1}\n"
            "]";
    }
    
    SetupLayout() {
        // Customized layout for 2D classification
        Splitter h_split;
        h_split.Horz();
        h_split << control_panel << data_view_panel;
        
        // Add network view to the data visualization panel
        data_view_panel.Add(data_view.VSizePos().HSizePos());
        data_view_panel.Add(bottom_controls.BottomPos(0, 60).HSizePos());
        
        Add(h_split.SizePos());
    }
}
```

## Key Design Principles

1. **Unified Window Base**: Either TopWindow or DockWindow with common control patterns
2. **Session Management**: Background training with pause/resume/stop controls
3. **JSON Network Configuration**: Editable network architecture
4. **Visualization Components**: Various visualizers for different aspects of neural networks
5. **Layout Systems**: Both splitter-based and docking window layouts
6. **Multi-AI Support**: Framework for visualizing multiple AI instances simultaneously
7. **Iterative Update Pattern**: Background training with UI refresh callbacks

The design allows for easy extension to create new examples while maintaining common functionality and UI patterns.