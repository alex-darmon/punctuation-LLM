# position_matched_detection_v3 — PREVIEW RUN

`run_position_matched_detection.py --config campaigns/position_matched_detection_v3.json`,
run 2026-09-19 in the same validated sandbox as `inference_v3_preview` (see that
directory's PROVENANCE_PREVIEW.md for the environment substitutions and the byte-for-byte
reproduction of inference_v2 that validates them). Re-run on the Mac for the canonical
manifest. The config is identical to `position_matched_detection.json` except that
`inference_config` points at the v3 declaration and `output_dir` at this directory.

Purpose: the manuscript's §4 cites equal-length prefix-vs-late 2,000-mark attribution for
the repeated-prompt runs (Flash 13.5% → 10.0%, Pro 16.0% → 10.0%). No artefact for that
existed — `position_matched_detection/` covers the standard-protocol batch only. This run
supplies it. Detection rows reproduce the campaign's `detection_position.csv` exactly.

The manifest here was written after stage 1 (prefix vs late); the five-window stage was
still running when these files were copied.
