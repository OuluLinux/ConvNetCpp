# basic1: Scientific Document Example Dataset

This directory is a scientific-literature example project that mirrors the same user-file surface as `engine_provider_a_8p`:

- `basic1.annsln` (project wiring + runtime options + cv_template_groups)
- `basic1.annlay` (slot/group model and training semantics)
- `basic1.annprj` (datasets/images/annotations/meta)
- `basic1_recognition.py` (metadata projection script)
- `basic1.mlui` (layout editor reference)

## Feature coverage mirrored from engine_provider_a_8p

- Pass-based model sets (`pass1`/`pass2`) in `.annsln`
- `cv_template_groups` in `.annsln`
- Composite slots (`composite_type: "token"`, `split_policy`, `sub_groups`, `sub_heads`) in `.annlay`
- Explicit `groups[]` for bool + label heads in `.annlay`
- OCR slot in `.annlay` (`doc_title`)
- Bool gates in `.annlay` (`figure_gate`, `equation_gate`, `caption_gate`)
- Recognition script metadata writes in `basic1_recognition.py`
- Verified metadata channel in `.annprj` (`metadata_verified`)

## Quick smoke checks

```bash
bin/AnnotationEditor --dump-project-manager datasets/basic1/basic1.annsln
bin/AnnotationEditor --export-crops datasets/basic1/basic1.annsln --pass 1
```
