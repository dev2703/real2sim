# Model provenance

Menagerie source: https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74

- SO-101: `menagerie/robotstudio_so101/LICENSE` (Apache-2.0); upstream derivation and
  contributors remain in its README and changelog.
- Panda: `menagerie/franka_emika_panda/LICENSE`; see its upstream README for attribution.
- Upstream XML, robot collision geometry, actuators and meshes are unmodified.
- `menagerie/manifest.json` pins every downloaded file with size, Git blob ID and SHA-256.
  Reproduce with `uv run python scripts/fetch_menagerie.py`.
- Generated task primitives (table, mat, ball, frustum bowl, camera) are local project additions
  assembled in memory by `real2sim.sim.model`. Their dimensions and provenance live in
  `configs/task.yaml` (labels in `docs/calibration.md`); simulator knobs are in
  `configs/sim_so101.yaml`. They form a provisional virtual scene, not a calibrated digital twin.

Upstream preview images are omitted. Panda assets are pinned for M5; a Panda task environment
has not been implemented by M2.
