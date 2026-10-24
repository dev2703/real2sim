# Proprioception noise audit

Inner-training recordings only: 22 episodes, 25355 raw rows.
Low-motion windows: 4803 (19.2%). These windows overlap and are proxies for stationarity.
Maximum arm steps (degrees): [2.6374, 3.1648, 2.8132, 3.3407, 2.6374].
Isolated return spikes >1 degree: [0, 0, 0, 0, 0].
Episode-median detrended low-motion RMS (degrees): [0.01382, 0.00812, 0.00975, 0.01416, 0.01282].
Nonuniform nominal timestamp increments: 0.

These metrics do not establish calibration or actual motor/camera acquisition timing. High-frequency power includes quantization and legitimate motion.
EMA new-sample weights 0.5 / 0.75 introduce theoretical low-frequency delays of 100 / 33 ms at 10 Hz. Their utility must be judged by matched active/phase prediction and physical tracking, not by variance reduction alone.
No raw sample, Cartesian target, gripper event or safety threshold was altered; the experiment variants change policy observation features only.
