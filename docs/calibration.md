# Calibration and physical measurements

**Status:** nothing measured yet. This file lists what is required, who must supply it, and the
provisional values that will be used (clearly labelled) until the measurement arrives. Nothing
here may be cited as a calibrated digital twin.

## Measurements requested from the operator

| Item | Needed for | Provisional value until measured | Status |
|---|---|---|---|
| Tennis ball diameter, mass | ball collision geom | ITF standard 6.54–6.86 cm, 56.0–59.4 g | UNKNOWN |
| Bowl interior radius at rim, interior depth, rim thickness, outer height, mass | compound bowl collision geometry | none; sim bowl is placeholder until measured | UNKNOWN |
| Black mat size and position relative to the robot base plate | task frame origin, start region | none | UNKNOWN |
| Ball start region (the paper marker seen in the iphone view): its size and position on the mat, and how much the start pose varied across episodes | reset distribution, scenario seeds | none | UNKNOWN |
| Bowl position relative to the robot base, and whether it moved between episodes | goal pose | none | UNKNOWN |
| Robot base plate position on the table; table height | world and task frames | none | UNKNOWN |
| iPhone camera: height above table, distance from robot base, tilt, which iPhone model and which lens (field of view) | simulated camera extrinsics and intrinsics | none; visual matching only | UNKNOWN |
| Gripper: jaw opening (mm) at gripper state 0, 50 and 100 | gripper semantic mapping and sim ctrl range | none | UNKNOWN |
| Recording command used (confirms `use_degrees`, cameras, fps) | unit assumptions | assumed degrees | UNKNOWN |

## Calibration procedure (to execute at M2, not yet done)

1. Real joint conventions: compare `observation.state` extremes against the Menagerie
   `robotstudio_so101` joint ranges and the operator's calibration files
   (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/aligned_follower.json`) to
   establish sign and zero per joint. Verify by FK on three distinctive dataset poses against
   the iphone frames at the same timestamps.
2. Camera: estimate iphone extrinsics by solving PnP on known robot geometry (base plate corners,
   FK-predicted gripper positions at several frames) and the bowl rim ellipse. Report reprojection
   error. Phone height plus "about 45° tilt" is not sufficient on its own.
3. Record all transforms as `T_parent_child`, metres, radians, right-handed, in
   `configs/task.yaml` with their provenance (measured / estimated / provisional).
