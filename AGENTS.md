# ConvNetCpp Agent Guide

## Repo Role

ConvNetCpp is a scientific engine, GUI, and CLI repository for machine learning and recognition tasks. The companion dataset repository is at `../`.

Build command:
```
script/build.py -r -m 8 -j12 <Target>
```

## Scientific Terminology Mandate
To maintain architectural purity and domain-agnostic scalability, always use generic, scientific terminology in code, commit messages, and documentation.
- **Element**: Primary recognized object unit.
- **Category**: Classification grouping for elements.
- **Level**: Scalar or ordinal value associated with an element.
- **GameEngine**: The host pipeline or simulation environment.
- **ProviderA**: The specific data source or upstream provider.
- **DS Repo**: Reference to the dataset repository.

## Architecture
- `src/AnnLayCore`: Core logic for annotation layers and generic recognition pipelines.
- `src/AnnotationEditor`: Main GUI tool for dataset management and model training.
- `src/ConvNet`: Primary neural network engine.
- `src/OCR`: Optical Character Recognition engine and utilities.

## Development Workflow
1. Use the `AnnotationEditor` branch for all active development.
2. Maintain compatibility with existing datasets via the `GroupRegistry` alias system.
3. Validate parity by running `bin/AnnotationEditor --frame-recognizer-dump-steps` against reference images in the dataset repository.
4. Ensure all commit messages adhere to scientific and professional standards.

## Task Lifecycle & Policy
- **Phase Completion**: At the end of a project phase, mark all completed tasks as 'done' in any associated JSON tracking stores.
- **Structural Integrity**: Preserve phase structure and numbering. Never leave completed tasks marked as 'todo'.
- **Naming Conventions**: Phase IDs should be non-numeric and include a track prefix (e.g., `umk1`).
- **Engine Rules**: Respect engine enablement matrices (planner vs. worker roles) when producing structured plans.
