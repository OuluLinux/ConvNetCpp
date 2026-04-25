# OCRPlugin

## Purpose
`OCRPlugin` adds image-based verification to autonomous game development workflows.

The plugin exists so an AI agent trainer can validate what is actually rendered on screen, not only internal runtime state. In iterative generation loops, a blind agent can produce outputs that are internally coherent but visually invalid. `OCRPlugin` is the visual-truth verifier that reads rendered frames and feeds validation results back to the trainer.

## Scope
- Integrate with ScriptIDE plugin lifecycle (no standalone `GUI_APP_MAIN`).
- Read active `.gamestate` and `.form` context from the host IDE.
- Resolve layout zones to concrete pixel rectangles.
- Run OCR/classification on zone crops using `src/OCR`.
- Expose verification API to ByteVM/Python for automated training loops.

## Existing Code References
- ScriptIDE plugin contracts:
  - `../ai-upp/uppsrc/ScriptIDE/PluginInterfaces.h`
  - `../ai-upp/uppsrc/ScriptIDE/PluginManager.h`
  - `../ai-upp/uppsrc/ScriptIDE/PythonIDE.cpp`
- ScriptIDE reference runtime host for `.gamestate` + `.form`:
  - `../ai-upp/uppsrc/ScriptIDE/CardGamePlugin.h`
  - `../ai-upp/uppsrc/ScriptIDE/CardGamePlugin.cpp`
- OCR engine and pipeline:
  - `../ConvNetCpp/src/OCR/OCR.h`
  - `../ConvNetCpp/src/OCR/OCRPipeline.h`
  - `../ConvNetCpp/src/OCR/API.md`

## Planned Public Surface
- `ocr_verify.capture_frame()`
- `ocr_verify.read_zone(zone_id)`
- `ocr_verify.read_zones(zone_ids)`
- `ocr_verify.compare(expected_state)`
- `ocr_verify.report_schema()`
- `ocr_verify.last_report()`

## Registration Model
Package is host-embedded and self-registers from an `.icpp` via `INITBLOCK { ... }`.

## `.gamestate` Metadata Contract
`OCRPlugin` reads the following optional keys from `.gamestate.metadata`:

- `ocr_verify`: enable OCRPlugin execute routing for this `.gamestate`.
- `ocr_expected`: map of `{zone_id: expected_text}` used for visual truth checks.
- `ocr_min_confidence`: numeric threshold (0.0..1.0) for warning on low-confidence OCR reads.

## Verification Report Contract
Reports returned by `compare()` and execute-mode verification follow:

- `schema`: `OCRVerificationReport`
- `schema_version`: `1.0`
- `signal`: one of `pass`, `warn`, `fail`
- `pass`: convenience boolean (`signal == "pass"`)
- `reason_codes`: unique list of reason codes from all zones
- `zones`: per-zone rows with:
  - `zone_id`, `ok`, `text`, `confidence`, `rect`
  - `expected` (when provided), `observed`, `match`
  - `severity` (`pass|warn|fail`), `reason_code`, `hint`

Current reason codes:
- `OCR_OK`
- `OCR_READ_ERROR`
- `OCR_TEXT_MISMATCH`
- `OCR_LOW_CONFIDENCE`
