# QWEN.md - ConvNetCpp Development Notes

This file contains development notes and pointers to other important documentation files.

## Documentation References

For information about development agents, see: [AGENTS.md](AGENTS.md)

For current development tasks, see: [TASKS.md](TASKS.md)

For repository analysis, see: [docs/REPOSITORY_ANALYSIS.md](docs/REPOSITORY_ANALYSIS.md)

For git history analysis, see: [docs/GIT_HISTORY.md](docs/GIT_HISTORY.md)

For UML diagrams, see: [UML_DIAGRAMS.md](UML_DIAGRAMS.md)

## Notes
- Always refer to AGENTS.md for information about specialized development agents
- Check TASKS.md for current development priorities
- Repository analysis and git history are documented in the docs/ folder
- UML diagrams and architecture documentation are in UML_DIAGRAMS.md

## Commit Rules
- QWEN cannot be marked as a co-author in git commits

## Coding Conventions
- Use U++ conventions: std::unique_ptr translates to Upp::One<> + Upp::Ptr<> + Upp::Pte<>. No std classes unless required because of accelerators.