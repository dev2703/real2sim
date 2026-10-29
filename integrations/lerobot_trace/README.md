# Local collector trace

Installed on17 October2026 in the sibling `../lerobot` checkout. The existing user change that records `_sent_action` was preserved. The optional `--recording_trace_dir` flag enables buffered sidecars with actual host timing, leader requests, processed requests, returned sent targets and measured follower positions. Defaults, hardware commands, motor reads and safety settings remain unchanged.

`recording_trace.py` is the maintained helper source. `prepare_patch.py` stages the recorder integration against the original collector context without applying it. It deliberately fails if those contexts have changed; do not rerun it on an already patched recorder. It requires the existing `_sent_action` fix. The original and patched snapshots, diff and hashes for this installation are in `outputs/collection_trace_patch/`. The helper and installed recorder were verified byte-for-byte against these staged sources.

`tests/unit/test_recording_trace_integration.py` exercises buffering and capture outcomes, and runs the actual installed or staged recording loop against fake hardware. `tests/unit/test_recording_trace_audit.py` checks physical-time gaps, delayed feedback, stable-command segmentation and label verification. The collector integration tests skip if neither optional local recorder artifact exists; the helper and auditor tests remain independent of hardware.

Pilot setup and limitations: [collection findings](../../docs/PROJECT_SUMMARY.md). Read-only analysis: `scripts/audit_recording_trace.py`. Camera exposure times and frame IDs are not supplied by this interface. No hardware timing overhead or real trace has yet been measured.
