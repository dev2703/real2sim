# New25FPS demonstration: read-only review

Source: `/Users/deveshvyas/.cache/huggingface/lerobot/local/so101_act_pickplace_v2_25fps`, episode0. This is separate from the40-episode30FPS source used by the active architecture queue. Source file hashes and the executed diagnostic are retained alongside this report. No raw data, calibration, controller setting or safety threshold was changed.

## Recording integrity and visible task

-467 data rows and467 decoded frames in each camera. Video PTS agree with nominal row timestamps to within0.92microseconds. No exactly identical consecutive160×90 grayscale frames were found; this does not rule out camera frame reuse obscured by encoding or missing physical samples.
- Nominal stored span is18.64s; encoded duration is18.68s. Actual physical sampling intervals are unavailable. The directory has no wall-clock trace, acquisition timestamps or separate requested/sent command traces. The reported23.69Hz benchmark cannot be reconstructed from this dataset alone.
- The phone-camera contact sheet shows grasp, lift/transport and release into the bowl; subsequent sampled frames show the ball remaining in the bowl. This is useful visual evidence of a successful real demonstration. It is not a simulation or learned-policy pass.
- Arm values are consistent with the current motor library's degrees quantization; the dataset itself does not explicitly store `use_degrees` or the collector commit. The local recorder writes returned sent targets, consistent with the user's account, but acquisition provenance should still be stored for future recordings.

## Tracking findings

Same-row target-minus-position includes ordinary response delay because observations precede sending. It must not be read as steady-state servo error. Mean absolute gaps across this episode are1.08,3.56,2.48,2.71,0.54degrees for the five arm joints; the largest observed arm gap is7.20degrees at shoulder lift. Thus the60–80degree leader-versus-safe warnings are not60–80degree recorded-target-versus-follower discrepancies in this episode.

To isolate stable targets, select overlapping windows spanning one nominal second with target peak-to-peak <=1degree (or1gripper percentage point), then compare the final measured position to the window's median target. These are diagnostic windows, not independent trials or new acceptance thresholds.

| Joint/phase | Nominal endpoint interval | Recorded target median | Measured median | Persistent signed target−measured gap |
|---|---|---:|---:|---:|
| Shoulder lift, initial hold |1.00–2.88s|−104.70°|−104.44°|−0.18°|
| Shoulder lift, later hold |6.84–7.44s|31.47°|32.18°|−0.70°|
| Shoulder lift, grasp/transport interval |8.60–11.60s|26.65°|31.65°|**−5.00°**|
| Elbow, transport interval |10.20–11.44s|5.85°|9.01°|**−3.16°**|
| Wrist flex, transport interval |10.32–10.92s|−30.64°|−29.49°|−1.14°|
| Gripper, ball grasp |7.56–11.20s|34.54%|37.09%|−2.56percentage points|

The shoulder's measured range during the8.60–11.60s interval is only0.70degree. This persistent gap warrants a loaded-hold diagnostic and cannot be dismissed as merely comparing one pre-send observation to a new target. The residual also resembles a5degree relative-command cap, but absent raw requested targets and synchronized command/readback traces, its cause is unresolved. Do not conclude motor damage, change gains, relax the limiter or rewrite labels from this episode alone. Gripper residual while holding the ball can be expected contact resistance, so measured opening should not replace the commanded grip target automatically.

The descriptive moving-data lag minimum occurs at three nominal frames for all six joints. That is120ms in stored time, not established physical latency; it is an in-sample descriptive minimum and does not justify a global label shift.

## Geometry and conversion implications

Under the current provisional mapping,111 recorded shoulder rows lie outside the model's native−100degree lower joint limit (the recording reaches−104.70degrees). That is a model/mapping compatibility issue, not proof the real robot violated its own limits. Another63 elbow rows lie just beyond the model's upper limit by roughly0.005degree, a tiny boundary discrepancy. Limits remain unchanged. This new episode can support geometry validation after camera/table/ball registration; applying the old scene calibration blindly is not valid.

The existing converter requires an integer source/policy rate ratio.25/10=2.5, so this episode cannot use the30→10 every-third-row conversion. Keep it separate. A new conversion must specify its timing assumptions, image sampling and action semantics; actual acquisition timing is preferable, and long stalls cannot be reconstructed from nominal timestamps.

## Next experiment

Use a short synchronized hold/approach/grasp trace with requested targets, sent targets, subsequent measured positions and monotonic read/send times. Include unloaded and ordinary ball-loaded holds at comparable poses, using the existing authorized controller and limits. This can distinguish persistent load-dependent tracking gaps from limiting, timestamp alignment and calibration issues. No hardware action was initiated in this audit. Continue the existing bounded architecture queue, but prioritize trace/provenance and geometry verification before extending training budgets.

Artifacts: `report.json`, `followup.json`, `joint_traces.csv`, `iphone_contact_sheet.jpg`, `top_contact_sheet.jpg`, and `executed_script.py`.
